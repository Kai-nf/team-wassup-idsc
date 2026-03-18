import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pywt
import wfdb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from wfdb import processing as wfproc

from environment_setup.data_loader import load_raw_dataset


FS = 100  # Hz, sampling frequency
WINDOW_SECONDS = 1.0
SAMPLES_BEFORE = int(0.5 * FS)  # 0.5 s before R
SAMPLES_AFTER = int(0.5 * FS)   # 0.5 s after R
WINDOW_SAMPLES = SAMPLES_BEFORE + SAMPLES_AFTER + 1  # 101
R_PEAK_MERGE_TOL = int(0.2 * FS)  # +/- 0.2s -> +/- 20 samples


# Hard-coded list of noisy leads discovered during Phase 1
NOISY_LEADS_BY_PATIENT: Dict[str, List[str]] = {
    "267630": ["I"],
    "287355": ["aVL"],
    "519139": ["III"],
    "801261": ["V2", "V5"],
    "930107": ["III"],
    "999474": ["aVL"],
    "1088175": ["aVL"],
    "1141322": ["III", "aVF"],
    "1142915": ["V3", "V4", "V5", "V6"],
    "1180702": ["aVL"],
    "1230482": ["I", "V2", "V3", "V5", "V6"],
    "1275329": ["V1"],
    "1313974": ["III"],
    "3058024": ["III", "aVL", "aVF", "V3"],
}

# Assumed canonical lead order for loaded NumPy arrays
STANDARD_LEAD_ORDER: List[str] = [
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]


def _find_record_prefix(records_path: Path, patient_id: int) -> Path:
    """
    Locate the WFDB record prefix for a given patient ID under `records_path`.

    This is robust to different layouts, e.g.:
      - records_path / "<patient_id>.hea"
      - records_path / "<patient_id>/<patient_id>.hea"
      - any deeper nested directory such as records_path / "files/<patient_id>/<patient_id>.hea"

    Returns the prefix path WITHOUT extension (suitable for wfdb.rdsamp),
    or raises FileNotFoundError if no matching .hea file is found.
    """
    pid_str = str(patient_id)

    # First, try the most common patterns quickly
    direct_hea = records_path / f"{pid_str}.hea"
    if direct_hea.exists():
        return direct_hea.with_suffix("")

    subdir_hea = records_path / pid_str / f"{pid_str}.hea"
    if subdir_hea.exists():
        return subdir_hea.with_suffix("")

    # Fallback: recursive search anywhere under records_path
    for hea_path in records_path.rglob(f"{pid_str}.hea"):
        return hea_path.with_suffix("")

    raise FileNotFoundError(
        f"No WFDB header file found for patient {patient_id} under {records_path}"
    )


def rescue_noisy_leads(
    raw_signal: np.ndarray,
    lead_names: List[str],
    patient_id: int,
) -> np.ndarray:
    """
    Phase 1 data rescue using Einthoven's law and Goldberger's equations.

    - raw_signal: shape (T, 12), as returned by wfdb (time, leads).
    - lead_names: ordered list of lead names corresponding to columns in raw_signal.
    - patient_id: integer patient identifier.

    Logic:
    - If patient is in NOISY_LEADS_BY_PATIENT, determine which LIMB leads are flagged.
    - For flagged LIMB leads (I, II, III, aVR, aVL, aVF), overwrite that column using
      analytical relationships:
         III = II - I
         aVR = -(I + II) / 2
         aVL = (I - III) / 2
         aVF = (II + III) / 2
      using Leads I and II as baseline truth whenever possible.
    - If Lead I itself is flagged, reconstruct it as I = II - III when III is available.
    - For flagged precordial leads (V1–V6), do NOT reconstruct; leave as-is to be
      handled by subsequent wavelet denoising.
    """
    pid_str = str(patient_id)
    if pid_str not in NOISY_LEADS_BY_PATIENT:
        return raw_signal

    noisy_leads = set(NOISY_LEADS_BY_PATIENT[pid_str])
    rescued = raw_signal.copy()

    # Map lead names to column indices for convenience
    name_to_idx: Dict[str, int] = {name: i for i, name in enumerate(lead_names)}

    limb_leads = ["I", "II", "III", "aVR", "aVL", "aVF"]
    present_limb = [l for l in limb_leads if l in name_to_idx]

    if not present_limb:
        # Non-standard layout; nothing we can safely do.
        return rescued

    # Extract limb lead time series (where available)
    limb_series: Dict[str, np.ndarray] = {
        l: rescued[:, name_to_idx[l]] for l in present_limb
    }

    # Step 1: If I is noisy but II and III are available and not noisy,
    # reconstruct I = II - III.
    if "I" in noisy_leads and "II" in limb_series and "III" in limb_series:
        I_new = limb_series["II"] - limb_series["III"]
        limb_series["I"] = I_new
        rescued[:, name_to_idx["I"]] = I_new
        noisy_leads.discard("I")

    # Step 2: If we have I and II (either originally clean or reconstructed),
    # use them as baseline truth to derive the other limb leads.
    if "I" in limb_series and "II" in limb_series:
        I = limb_series["I"]
        II = limb_series["II"]

        III_new = II - I
        aVR_new = -(I + II) / 2.0
        aVL_new = (I - III_new) / 2.0
        aVF_new = (II + III_new) / 2.0

        reconstructed = {
            "III": III_new,
            "aVR": aVR_new,
            "aVL": aVL_new,
            "aVF": aVF_new,
        }

        for lead_name, series in reconstructed.items():
            # Only overwrite limb leads explicitly flagged as noisy.
            if lead_name in noisy_leads and lead_name in name_to_idx:
                rescued[:, name_to_idx[lead_name]] = series

    # Precordial leads (V1–V6) are intentionally left untouched, even if flagged.
    return rescued


def wavelet_denoise_lead(signal_1d: np.ndarray) -> np.ndarray:
    """
    Apply Daubechies 4 (db4) wavelet denoising with:
    - 4-level decomposition
    - Zeroed Level-4 approximation coefficients (cA4) for baseline wander removal
    - Universal soft thresholding applied to all detail coefficients.
    """
    wavelet = "db4"
    max_level = 4
    coeffs = pywt.wavedec(signal_1d, wavelet, level=max_level)

    # coeffs[0] is cA4 for level=4. Zero it to remove baseline wander.
    cA4 = coeffs[0]
    coeffs[0] = np.zeros_like(cA4)

    # Apply universal soft thresholding to detail coefficients at each level.
    # coeffs[1:] are detail coefficients [cD4, cD3, cD2, cD1]
    n = len(signal_1d)
    for i in range(1, len(coeffs)):
        cD = coeffs[i]
        # Robust noise estimate using MAD
        if cD.size == 0:
            continue
        sigma = np.median(np.abs(cD)) / 0.6745
        if sigma <= 0:
            continue
        threshold = sigma * np.sqrt(2 * np.log(n))
        coeffs[i] = pywt.threshold(cD, threshold, mode="soft")

    denoised = pywt.waverec(coeffs, wavelet)

    # Ensure the reconstructed signal has the original length
    if denoised.shape[0] > signal_1d.shape[0]:
        denoised = denoised[: signal_1d.shape[0]]
    elif denoised.shape[0] < signal_1d.shape[0]:
        pad = signal_1d.shape[0] - denoised.shape[0]
        denoised = np.pad(denoised, (0, pad), mode="edge")

    return denoised


def apply_wavelet_denoising(signal: np.ndarray) -> np.ndarray:
    """
    Apply wavelet_denoise_lead independently to each of the 12 leads.

    signal: shape (T, L) where L=12.
    """
    T, L = signal.shape
    denoised = np.zeros_like(signal)
    for lead_idx in range(L):
        denoised[:, lead_idx] = wavelet_denoise_lead(signal[:, lead_idx])
    return denoised


def _detect_r_peaks_single_lead(lead_sig: np.ndarray, fs: int) -> np.ndarray:
    """
    Wrapper around WFDB's XQRS detector that is compatible with different
    wfdb versions.

    - If `wfdb.processing.xqrs_detect` exists, use it directly.
    - Otherwise, fall back to the `wfdb.processing.XQRS` class.
    """
    # Newer wfdb versions expose a convenience function
    if hasattr(wfproc, "xqrs_detect"):
        return wfproc.xqrs_detect(sig=lead_sig, fs=fs)

    # Fallback: use the XQRS class explicitly
    if hasattr(wfproc, "XQRS"):
        xqrs = wfproc.XQRS(sig=lead_sig, fs=fs)
        xqrs.detect()
        return np.asarray(xqrs.qrs_inds, dtype=int)

    raise RuntimeError("wfdb processing does not provide XQRS detection APIs.")


def detect_multi_lead_consensus_r_peaks(
    denoised_signal: np.ndarray,
    fs: int,
    min_leads: int = 3,
) -> np.ndarray:
    """
    Multi-lead consensus R-peak detection:
    - Apply XQRS detection per lead.
    - Merge candidate peaks across leads within +/- R_PEAK_MERGE_TOL samples.
    - Discard groups with detections from < 3 distinct leads.
    - Use the integer mean of contributing peaks as the consensus index.

    denoised_signal: shape (T, L)
    Returns: 1D array of consensus R-peak indices.
    """
    T, L = denoised_signal.shape
    merge_tol = int(0.2 * fs)  # +/- 0.2s in samples

    # Collect all candidate peaks with their originating lead indices
    candidate_peaks: List[Tuple[int, int]] = []  # (sample_index, lead_index)

    for lead_idx in range(L):
        lead_sig = denoised_signal[:, lead_idx]
        try:
            r_peaks = _detect_r_peaks_single_lead(lead_sig, fs=fs)
        except Exception:
            # If XQRS fails on a particular lead, skip it.
            continue
        for r in r_peaks:
            if 0 <= r < T:
                candidate_peaks.append((int(r), lead_idx))

    if not candidate_peaks:
        return np.array([], dtype=int)

    # Sort by sample index
    candidate_peaks.sort(key=lambda x: x[0])

    # Group peaks within +/- R_PEAK_MERGE_TOL samples
    groups: List[List[Tuple[int, int]]] = []
    current_group: List[Tuple[int, int]] = [candidate_peaks[0]]

    for pk, ld in candidate_peaks[1:]:
        last_pk = current_group[-1][0]
        if abs(pk - last_pk) <= merge_tol:
            current_group.append((pk, ld))
        else:
            groups.append(current_group)
            current_group = [(pk, ld)]
    groups.append(current_group)

    consensus_peaks: List[int] = []
    for group in groups:
        sample_indices = np.array([g[0] for g in group], dtype=int)
        lead_indices = np.array([g[1] for g in group], dtype=int)
        unique_leads = np.unique(lead_indices)

        # Require detections from at least `min_leads` distinct leads
        if unique_leads.size < min_leads:
            continue

        consensus_idx = int(np.round(sample_indices.mean()))
        consensus_peaks.append(consensus_idx)

    return np.array(sorted(set(consensus_peaks)), dtype=int)


def segment_beats_around_r_peaks(
    signal: np.ndarray,
    r_peaks: np.ndarray,
) -> List[np.ndarray]:
    """
    Extract fixed 1-second windows centered on each R-peak:
    - Uses [peak - SAMPLES_BEFORE : peak + SAMPLES_AFTER + 1]
      which yields shape (WINDOW_SAMPLES, L) = (101, 12).
    - Discards beats that would exceed the array boundaries.
    """
    T, L = signal.shape
    beats: List[np.ndarray] = []

    for r in r_peaks:
        start = r - SAMPLES_BEFORE
        end = r + SAMPLES_AFTER + 1  # exclusive index
        if start < 0 or end > T:
            continue
        beat = signal[start:end, :]
        if beat.shape[0] == WINDOW_SAMPLES:
            beats.append(beat)

    return beats


def build_method3_wavelet_dataset(
    records_dir: str,
    metadata_csv_path: str = "metadata.csv",
    output_dir: str = "data_preprocessing",
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[np.ndarray, Dict]:
    """
    Method 3 (Advanced Signal Processing) end-to-end pipeline.

    Parameters
    ----------
    records_dir:
        Directory containing WFDB records. Each patient is expected to have
        `<patient_id>.dat` and `<patient_id>.hea` stored here, so that
        `wfdb.rdsamp(Path(records_dir) / str(patient_id))` succeeds.
    metadata_csv_path:
        Path to `metadata.csv` containing at least `patient_id` and `brugada`.
    output_dir:
        Directory where `dataset_v3_wavelet.npy` and `fold_composition_v3.json`
        will be written.
    n_splits:
        Number of StratifiedGroupKFold splits (default 5).
    random_state:
        Random seed for reproducibility of cross-validation splits.

    Returns
    -------
    X_beats : np.ndarray
        Unscaled beat-level dataset of shape (Total_Beats, 101, 12).
    fold_metadata : dict
        Dictionary mirroring the contents of `fold_composition_v3.json`.
    """
    records_path = Path(records_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(metadata_csv_path)

    if "patient_id" not in metadata.columns or "brugada" not in metadata.columns:
        raise ValueError("metadata.csv must contain 'patient_id' and 'brugada' columns.")

    # Binary labels: 1 if any form of Brugada is present (brugada > 0), else 0.
    # This collapses values {1, 2} into the positive class.
    metadata = metadata.copy()
    metadata["label"] = (metadata["brugada"].fillna(0) > 0).astype(int)

    all_beats: List[np.ndarray] = []
    all_labels: List[int] = []
    all_patient_ids: List[int] = []

    for _, row in metadata.iterrows():
        patient_id = int(row["patient_id"])
        label = int(row["label"])

        try:
            record_prefix = _find_record_prefix(records_path, patient_id)
            record = wfdb.rdrecord(str(record_prefix))
        except Exception as e:
            # If a record is missing or unreadable, skip it but warn.
            print(f"Warning: could not read record for patient {patient_id}: {e}")
            continue

        signal = record.p_signal  # (T, L)
        lead_names = record.sig_name
        fs_record = int(round(getattr(record, "fs", FS)))

        if signal.ndim != 2 or signal.shape[1] != 12:
            print(
                f"Warning: patient {patient_id} has unexpected signal shape {signal.shape}; skipping."
            )
            continue

        # Phase 1 rescue of noisy limb leads
        rescued = rescue_noisy_leads(signal, lead_names, patient_id=patient_id)

        # Wavelet denoising per lead
        denoised = apply_wavelet_denoising(rescued)

        # Multi-lead consensus R-peak detection on denoised signal
        r_peaks = detect_multi_lead_consensus_r_peaks(denoised, fs=fs_record)

        # Fallback: if strict consensus fails, try a more permissive strategy
        if r_peaks.size == 0:
            # First, relax the minimum-lead requirement
            r_peaks_relaxed = detect_multi_lead_consensus_r_peaks(
                denoised, fs=fs_record, min_leads=2
            )
            if r_peaks_relaxed.size > 0:
                print(
                    f"Warning: using 2-lead consensus R-peaks for patient {patient_id} "
                    f"(strict 3-lead consensus failed)."
                )
                r_peaks = r_peaks_relaxed
            else:
                # As a last resort, fall back to single-lead detection on lead II
                lead_idx_for_fallback = None
                if "II" in lead_names:
                    lead_idx_for_fallback = lead_names.index("II")
                else:
                    # If lead II is unavailable, just use the first lead
                    lead_idx_for_fallback = 0

                try:
                    r_peaks_single = _detect_r_peaks_single_lead(
                        denoised[:, lead_idx_for_fallback], fs=fs_record
                    )
                except Exception:
                    r_peaks_single = np.array([], dtype=int)

                if r_peaks_single.size > 0:
                    print(
                        f"Warning: falling back to single-lead R-peak detection "
                        f"for patient {patient_id} on lead index {lead_idx_for_fallback}."
                    )
                    r_peaks = r_peaks_single
                else:
                    print(
                        f"Warning: no reliable R-peaks detected for patient {patient_id}; skipping."
                    )
                    continue

        # Beat-level segmentation
        beats = segment_beats_around_r_peaks(denoised, r_peaks)
        if not beats:
            print(f"Warning: no valid beat windows for patient {patient_id}; skipping.")
            continue

        for beat in beats:
            all_beats.append(beat.astype(np.float32))
            all_labels.append(label)
            all_patient_ids.append(patient_id)

    if not all_beats:
        raise RuntimeError("No beats were extracted. Check WFDB paths and signal quality.")

    X_beats = np.stack(all_beats, axis=0)  # (Total_Beats, 101, 12)
    y_beats = np.array(all_labels, dtype=int)
    patient_ids = np.array(all_patient_ids, dtype=int)

    # Sanity check on window shape
    if X_beats.shape[1] != WINDOW_SAMPLES or X_beats.shape[2] != 12:
        raise ValueError(
            f"Unexpected beat shape {X_beats.shape[1:]}; expected ({WINDOW_SAMPLES}, 12)."
        )

    # StratifiedGroupKFold over beats, grouped by patient_id
    sgkf = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )

    fold_metadata: Dict[str, Dict] = {
        "n_splits": n_splits,
        "random_state": random_state,
        "labels": y_beats.tolist(),
        "patient_ids": patient_ids.tolist(),
        "folds": {},
    }

    for fold_idx, (train_idx, test_idx) in enumerate(
        sgkf.split(X_beats, y_beats, groups=patient_ids)
    ):
        fold_key = str(fold_idx)
        train_pids = sorted(set(patient_ids[train_idx].tolist()))
        test_pids = sorted(set(patient_ids[test_idx].tolist()))

        fold_metadata["folds"][fold_key] = {
            "train": {
                "patient_ids": train_pids,
                "beat_indices": train_idx.astype(int).tolist(),
            },
            "test": {
                "patient_ids": test_pids,
                "beat_indices": test_idx.astype(int).tolist(),
            },
        }

    # Save unscaled master array
    np.save(output_path / "dataset_v3_wavelet.npy", X_beats)

    # Save fold composition JSON
    with open(output_path / "fold_composition_v3.json", "w", encoding="utf-8") as f:
        json.dump(fold_metadata, f, indent=2)

    print(
        f"Method 3 complete. Saved dataset_v3_wavelet.npy "
        f"with shape {X_beats.shape} and fold_composition_v3.json."
    )

    return X_beats, fold_metadata


def build_method3_wavelet_dataset_from_loader(
    metadata_csv_path: str = "metadata.csv",
    data_dir: str = "files",
    output_dir: str = "data_preprocessing",
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[np.ndarray, Dict]:
    """
    Method 3 pipeline using the existing `load_raw_dataset` helper.

    This avoids any direct WFDB path handling here:
    - Uses `load_raw_dataset(csv_path, data_dir)` to obtain
      a (N, 1200, 12) array of raw signals and corresponding brugada labels.
    - Applies rescue, wavelet denoising, R-peak detection, and beat segmentation
      exactly as in `build_method3_wavelet_dataset`.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load raw signals and labels via the shared data loader
    all_signals, brugada_vals = load_raw_dataset(metadata_csv_path, data_dir)
    metadata = pd.read_csv(metadata_csv_path)

    if "patient_id" not in metadata.columns or "brugada" not in metadata.columns:
        raise ValueError("metadata.csv must contain 'patient_id' and 'brugada' columns.")

    if len(metadata) != all_signals.shape[0]:
        raise ValueError(
            f"Mismatch between metadata rows ({len(metadata)}) and "
            f"loaded signals ({all_signals.shape[0]})."
        )

    # Binary labels from metadata (more authoritative than raw brugada_vals)
    labels = (metadata["brugada"].fillna(0) > 0).astype(int).to_numpy()
    patient_ids = metadata["patient_id"].astype(int).to_numpy()

    all_beats: List[np.ndarray] = []
    all_labels: List[int] = []
    all_patient_ids: List[int] = []

    for idx in range(all_signals.shape[0]):
        patient_id = int(patient_ids[idx])
        label = int(labels[idx])
        signal = all_signals[idx]  # (T, 12)

        if signal.ndim != 2 or signal.shape[1] != 12:
            print(
                f"Warning: patient {patient_id} has unexpected signal shape {signal.shape}; skipping."
            )
            continue

        # Phase 1 rescue of noisy limb leads (assumed standard lead order)
        rescued = rescue_noisy_leads(signal, STANDARD_LEAD_ORDER, patient_id=patient_id)

        # Wavelet denoising per lead
        denoised = apply_wavelet_denoising(rescued)

        # Multi-lead consensus R-peak detection (with fallbacks)
        r_peaks = detect_multi_lead_consensus_r_peaks(denoised, fs=FS)

        if r_peaks.size == 0:
            r_peaks_relaxed = detect_multi_lead_consensus_r_peaks(
                denoised, fs=FS, min_leads=2
            )
            if r_peaks_relaxed.size > 0:
                print(
                    f"Warning: using 2-lead consensus R-peaks for patient {patient_id} "
                    f"(strict 3-lead consensus failed)."
                )
                r_peaks = r_peaks_relaxed
            else:
                # Single-lead fallback on lead II (index 1) when available
                lead_idx_for_fallback = 1 if signal.shape[1] > 1 else 0
                try:
                    r_peaks_single = _detect_r_peaks_single_lead(
                        denoised[:, lead_idx_for_fallback], fs=FS
                    )
                except Exception:
                    r_peaks_single = np.array([], dtype=int)

                if r_peaks_single.size > 0:
                    print(
                        f"Warning: falling back to single-lead R-peak detection "
                        f"for patient {patient_id} on lead index {lead_idx_for_fallback}."
                    )
                    r_peaks = r_peaks_single
                else:
                    print(
                        f"Warning: no reliable R-peaks detected for patient {patient_id}; skipping."
                    )
                    continue

        # Beat-level segmentation
        beats = segment_beats_around_r_peaks(denoised, r_peaks)
        if not beats:
            print(f"Warning: no valid beat windows for patient {patient_id}; skipping.")
            continue

        for beat in beats:
            all_beats.append(beat.astype(np.float32))
            all_labels.append(label)
            all_patient_ids.append(patient_id)

    if not all_beats:
        raise RuntimeError("No beats were extracted. Check input signals and detection settings.")

    X_beats = np.stack(all_beats, axis=0)
    y_beats = np.array(all_labels, dtype=int)
    pid_beats = np.array(all_patient_ids, dtype=int)

    if X_beats.shape[1] != WINDOW_SAMPLES or X_beats.shape[2] != 12:
        raise ValueError(
            f"Unexpected beat shape {X_beats.shape[1:]}; expected ({WINDOW_SAMPLES}, 12)."
        )

    sgkf = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )

    fold_metadata: Dict[str, Dict] = {
        "n_splits": n_splits,
        "random_state": random_state,
        "labels": y_beats.tolist(),
        "patient_ids": pid_beats.tolist(),
        "folds": {},
    }

    for fold_idx, (train_idx, test_idx) in enumerate(
        sgkf.split(X_beats, y_beats, groups=pid_beats)
    ):
        fold_key = str(fold_idx)
        train_pids = sorted(set(pid_beats[train_idx].tolist()))
        test_pids = sorted(set(pid_beats[test_idx].tolist()))

        fold_metadata["folds"][fold_key] = {
            "train": {
                "patient_ids": train_pids,
                "beat_indices": train_idx.astype(int).tolist(),
            },
            "test": {
                "patient_ids": test_pids,
                "beat_indices": test_idx.astype(int).tolist(),
            },
        }

    np.save(output_path / "dataset_v3_wavelet.npy", X_beats)
    with open(output_path / "fold_composition_v3.json", "w", encoding="utf-8") as f:
        json.dump(fold_metadata, f, indent=2)

    print(
        f"Method 3 complete (data_loader). Saved dataset_v3_wavelet.npy "
        f"with shape {X_beats.shape} and fold_composition_v3.json."
    )

    return X_beats, fold_metadata


def load_wavelet_dataset_with_master_manifest(
    fold_id: int,
    data_dir: str = "data_preprocessing",
    master_manifest_path: str = "master_folds_drop14.json"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads dataset_v3_wavelet.npy but splits it according to the patient-level 
    Master Manifest rather than the internal v3 beat-level folds.
    
    Returns scaled X_train, scaled X_test, y_train, y_test, and test_patient_ids.
    """
    data_path = Path(data_dir)
    X_path = data_path / "dataset_v3_wavelet.npy"
    v3_json_path = data_path / "fold_composition_v3.json" # Still needed to map beats to patients
    master_json_path = Path(master_manifest_path)

    if not X_path.exists(): raise FileNotFoundError(f"Missing {X_path}")
    if not v3_json_path.exists(): raise FileNotFoundError(f"Missing {v3_json_path}")
    if not master_json_path.exists(): raise FileNotFoundError(f"Missing {master_json_path}")

    # 1. Load the raw beats and the v3 beat-to-patient mapping
    X_beats = np.load(X_path)
    with open(v3_json_path, "r", encoding="utf-8") as f:
        v3_meta = json.load(f)
        
    all_labels = np.array(v3_meta["labels"], dtype=int)
    all_pids = np.array(v3_meta["patient_ids"], dtype=int)

    # 2. Load the Master Manifest patient assignments
    with open(master_json_path, "r", encoding="utf-8") as f:
        master_meta = json.load(f)
        
    fold_key = str(fold_id)
    if fold_key not in master_meta["folds"]:
        raise ValueError(f"Fold {fold_id} not found in {master_manifest_path}")

# We just need to go one level deeper into the JSON to grab the actual list
    master_train_pids = set(int(p) for p in master_meta["folds"][fold_key]["train"]["patient_ids"])
    master_test_pids = set(int(p) for p in master_meta["folds"][fold_key]["test"]["patient_ids"])

    # 3. Find which beat indices belong to the master train/test patients
    train_indices = [i for i, pid in enumerate(all_pids) if pid in master_train_pids]
    test_indices = [i for i, pid in enumerate(all_pids) if pid in master_test_pids]

    train_indices = np.array(train_indices, dtype=int)
    test_indices = np.array(test_indices, dtype=int)

    # 4. Slice the arrays
    X_train = X_beats[train_indices]
    X_test = X_beats[test_indices]
    y_train = all_labels[train_indices]
    y_test = all_labels[test_indices]
    test_pids = all_pids[test_indices]

    # 5. StandardScaler per lead (fit on train only)
    n_train, T, L = X_train.shape
    n_test = X_test.shape[0]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, L)).reshape(n_train, T, L)
    X_test_scaled = scaler.transform(X_test.reshape(-1, L)).reshape(n_test, T, L)

    return X_train_scaled.astype(np.float32), X_test_scaled.astype(np.float32), y_train, y_test, test_pids

if __name__ == "__main__":
    """
    Running the pipeline with hardcoded paths. 
    You can simply click 'Run' in your IDE now!
    """
    
    # --- HARDCODED VARIABLES ---
    # Make sure these match the actual folder and file names in your root directory
    METADATA_CSV = "metadata.csv"
    DATA_DIR = "files" 
    OUTPUT_DIR = "data_preprocessing"
    N_SPLITS = 5
    RANDOM_STATE = 42

    print("Starting Method 3 Pipeline with hardcoded variables...")
    print(f"Looking for ECG records in: {DATA_DIR}")
    print(f"Looking for metadata in: {METADATA_CSV}")

    build_method3_wavelet_dataset_from_loader(
        metadata_csv_path=METADATA_CSV,
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        n_splits=N_SPLITS,
        random_state=RANDOM_STATE,
    )
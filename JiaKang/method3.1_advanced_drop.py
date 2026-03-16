import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from pathlib import Path
from typing import Dict, List, Tuple


import numpy as np
import pandas as pd
import pywt
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from wfdb import processing as wfproc

from Environment_setup.data_loader import load_raw_dataset


FS = 100  # Hz, sampling frequency
WINDOW_SECONDS = 1.0
SAMPLES_BEFORE = int(0.5 * FS)  # 0.5 s before R
SAMPLES_AFTER = int(0.5 * FS)   # 0.5 s after R
WINDOW_SAMPLES = SAMPLES_BEFORE + SAMPLES_AFTER + 1  # 101


# Hard-coded list of noisy patients discovered during Phase 1
# These patients are strictly dropped in this high-purity pipeline.
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
    Wrapper around WFDB's XQRS detector that is compatible with different wfdb versions.

    - If `wfdb.processing.xqrs_detect` exists, use it directly.
    - Otherwise, fall back to the `wfdb.processing.XQRS` class.
    """
    if hasattr(wfproc, "xqrs_detect"):
        return wfproc.xqrs_detect(sig=lead_sig, fs=fs)

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
    - Merge candidate peaks across leads within +/- 0.2s.
    - Discard groups with detections from fewer than `min_leads` leads.
    - Use the integer mean of contributing peaks as the consensus index.

    denoised_signal: shape (T, L)
    """
    T, L = denoised_signal.shape
    merge_tol = int(0.2 * fs)

    candidate_peaks: List[Tuple[int, int]] = []  # (sample_index, lead_index)

    for lead_idx in range(L):
        lead_sig = denoised_signal[:, lead_idx]
        try:
            r_peaks = _detect_r_peaks_single_lead(lead_sig, fs=fs)
        except Exception:
            continue
        for r in r_peaks:
            if 0 <= r < T:
                candidate_peaks.append((int(r), lead_idx))

    if not candidate_peaks:
        return np.array([], dtype=int)

    candidate_peaks.sort(key=lambda x: x[0])

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
    - Uses [peak - SAMPLES_BEFORE : peak + SAMPLES_AFTER + 1],
      yielding shape (WINDOW_SAMPLES, L) = (101, 12).
    - Discards beats that would exceed the array boundaries.
    """
    T, L = signal.shape
    beats: List[np.ndarray] = []

    for r in r_peaks:
        start = r - SAMPLES_BEFORE
        end = r + SAMPLES_AFTER + 1
        if start < 0 or end > T:
            continue
        beat = signal[start:end, :]
        if beat.shape[0] == WINDOW_SAMPLES:
            beats.append(beat)

    return beats


def build_method3p1_wavelet_dataset_from_loader(
    metadata_csv_path: str = "metadata.csv",
    data_dir: str = "files",
    output_dir: str = "data_preprocessing",
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[np.ndarray, Dict]:
    """
    Method 3.1 (High-Purity Drop) pipeline using the existing `load_raw_dataset`.

    Changes relative to Method 3:
    - Patients listed in NOISY_LEADS_BY_PATIENT are strictly dropped.
    - No mathematical rescue of noisy limb leads; only denoising/filtering on
      the remaining clean patients.
    - Outputs:
        - dataset_v3.1_wavelet.npy
        - fold_composition_v3.1.json
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_signals, brugada_vals = load_raw_dataset(metadata_csv_path, data_dir)
    metadata = pd.read_csv(metadata_csv_path)

    if "patient_id" not in metadata.columns or "brugada" not in metadata.columns:
        raise ValueError("metadata.csv must contain 'patient_id' and 'brugada' columns.")

    if len(metadata) != all_signals.shape[0]:
        raise ValueError(
            f"Mismatch between metadata rows ({len(metadata)}) and "
            f"loaded signals ({all_signals.shape[0]})."
        )

    labels = (metadata["brugada"].fillna(0) > 0).astype(int).to_numpy()
    patient_ids = metadata["patient_id"].astype(int).to_numpy()
    noisy_patients = set(int(pid) for pid in NOISY_LEADS_BY_PATIENT.keys())

    all_beats: List[np.ndarray] = []
    all_labels: List[int] = []
    all_patient_ids: List[int] = []

    for idx in range(all_signals.shape[0]):
        patient_id = int(patient_ids[idx])

        if patient_id in noisy_patients:
            print(
                f"Warning: patient {patient_id} dropped due to high-frequency noise "
                f"flags to preserve signal purity."
            )
            continue

        label = int(labels[idx])
        signal = all_signals[idx]  # (T, 12)

        if signal.ndim != 2 or signal.shape[1] != 12:
            print(
                f"Warning: patient {patient_id} has unexpected signal shape {signal.shape}; skipping."
            )
            continue

        # Wavelet denoising per lead
        denoised = apply_wavelet_denoising(signal)

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

    np.save(output_path / "dataset_v3.1_wavelet.npy", X_beats)
    with open(output_path / "fold_composition_v3.1.json", "w", encoding="utf-8") as f:
        json.dump(fold_metadata, f, indent=2)

    print(
        f"Method 3.1 complete (drop noisy). Saved dataset_v3.1_wavelet.npy "
        f"with shape {X_beats.shape} and fold_composition_v3.1.json."
    )

    return X_beats, fold_metadata


def load_wavelet_dataset_for_fold(
    fold_id: int,
    data_dir: str = "data_preprocessing",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Helper function for the high-purity Method 3.1 dataset:
    - Loads `dataset_v3.1_wavelet.npy` and `fold_composition_v3.1.json`.
    - Applies per-lead StandardScaler fitted only on the training set for the
      requested fold.
    - Returns scaled train/test arrays and their labels.
    """
    data_path = Path(data_dir)
    X_path = data_path / "dataset_v3.1_wavelet.npy"
    json_path = data_path / "fold_composition_v3.1.json"

    if not X_path.exists():
        raise FileNotFoundError(f"Could not find {X_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"Could not find {json_path}")

    X_beats = np.load(X_path)
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    labels = np.array(meta["labels"], dtype=int)
    folds = meta["folds"]
    fold_key = str(fold_id)

    if fold_key not in folds:
        raise ValueError(f"Fold {fold_id} not found in fold_composition_v3.1.json")

    train_indices = np.array(folds[fold_key]["train"]["beat_indices"], dtype=int)
    test_indices = np.array(folds[fold_key]["test"]["beat_indices"], dtype=int)

    X_train = X_beats[train_indices]
    X_test = X_beats[test_indices]
    y_train = labels[train_indices]
    y_test = labels[test_indices]

    n_train, T, L = X_train.shape
    n_test = X_test.shape[0]

    scaler = StandardScaler()
    X_train_2d = X_train.reshape(-1, L)
    X_test_2d = X_test.reshape(-1, L)

    X_train_scaled_2d = scaler.fit_transform(X_train_2d)
    X_test_scaled_2d = scaler.transform(X_test_2d)

    X_train_scaled = X_train_scaled_2d.reshape(n_train, T, L)
    X_test_scaled = X_test_scaled_2d.reshape(n_test, T, L)

    return (
        X_train_scaled.astype(np.float32),
        X_test_scaled.astype(np.float32),
        y_train,
        y_test,
    )


if __name__ == "__main__":
    """
    Running the Method 3.1 pipeline with hardcoded paths. 
    You can simply click 'Run' in your IDE now!
    """
    
    # --- HARDCODED VARIABLES ---
    # Make sure these match the actual folder and file names in your root directory
    METADATA_CSV = "metadata.csv"
    DATA_DIR = "files" 
    OUTPUT_DIR = "data_preprocessing"
    N_SPLITS = 5
    RANDOM_STATE = 42

    print("Starting Method 3.1 Pipeline (High-Purity) with hardcoded variables...")
    print(f"Looking for ECG records in: {DATA_DIR}")
    print(f"Looking for metadata in: {METADATA_CSV}")

    build_method3p1_wavelet_dataset_from_loader(
        metadata_csv_path=METADATA_CSV,
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        n_splits=N_SPLITS,
        random_state=RANDOM_STATE,
    )

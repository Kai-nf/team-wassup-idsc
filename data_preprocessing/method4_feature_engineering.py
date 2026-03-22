"""
Method 4: Feature Engineering Preprocessing Pipeline
=====================================================
Extracts 19 clinically-motivated morphological features per patient
from 12-lead ECG signals for Brugada syndrome classification.
"""

import sys
import os
import json
import argparse
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Add project root to sys.path so `from data_loader import ...` works
# even when running this file directly from its subfolder.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Environment_setup.data_loader import load_raw_dataset  # (csv_path, data_dir) -> (N,1200,12), labels

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FS = 100

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_IDX = {name: i for i, name in enumerate(LEAD_NAMES)}

FEATURE_NAMES = [
    "st_elevation_v1",
    "st_elevation_v2",
    "st_elevation_v3",
    "qrs_duration_v1",
    "qrs_duration_v2",
    "qrs_duration_v3",
    "rj_interval_v1",
    "p_wave_duration_ii",
    "p_wave_duration_v1",
    "p_wave_amplitude_ii",
    "p_wave_amplitude_v1",
    "t_wave_amplitude_v1",
    "t_wave_amplitude_v2",
    "t_wave_amplitude_v3",
    "t_wave_symmetry_v2",
    "st_t_ratio_v2",
    "rr_mean_ii",
    "rr_std_ii",
    "heart_rate",
]

# ---------------------------------------------------------------------------
# Filtering (same logic as Method 2)
# ---------------------------------------------------------------------------
def apply_clinical_filters(signal: np.ndarray, fs: int = FS) -> np.ndarray:
    """Bandpass 0.5-40 Hz + Notch 50 Hz.  signal shape: (1200, 12)."""
    nyquist = 0.5 * fs
    low, high = 0.5 / nyquist, 40 / nyquist
    b, a = butter(3, [low, high], btype="band")
    filtered = filtfilt(b, a, signal, axis=0)
    b_n, a_n = iirnotch(50, 30, fs)
    filtered = filtfilt(b_n, a_n, filtered, axis=0)
    return filtered

# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------
def _safe_mean(arr):
    f = np.asarray(arr, dtype=float)
    f = f[np.isfinite(f)]
    return float(np.mean(f)) if f.size > 0 else np.nan

def _safe_std(arr):
    f = np.asarray(arr, dtype=float)
    f = f[np.isfinite(f)]
    return float(np.std(f, ddof=1)) if f.size > 1 else np.nan

def _get_indices(waves_dict, key):
    """Get valid integer sample indices from a NeuroKit2 delineation dict."""
    raw = waves_dict.get(key, [])
    if raw is None:
        return np.array([], dtype=int)
    arr = np.array(raw, dtype=float)
    valid = arr[np.isfinite(arr)]
    return valid.astype(int)

# ---------------------------------------------------------------------------
# Feature extraction – ONE patient
# ---------------------------------------------------------------------------
def extract_features_single_patient(
    signal_12lead: np.ndarray, fs: int = FS
) -> Dict[str, float]:
    """
    Extract 19 morphological features from a (1200, 12) filtered ECG.
    Returns dict keyed by FEATURE_NAMES; failed sub-features are np.nan.
    """
    features = {name: np.nan for name in FEATURE_NAMES}

    lead_ii = signal_12lead[:, LEAD_IDX["II"]]
    lead_v1 = signal_12lead[:, LEAD_IDX["V1"]]
    lead_v2 = signal_12lead[:, LEAD_IDX["V2"]]
    lead_v3 = signal_12lead[:, LEAD_IDX["V3"]]

    # ---- R-peak detection on Lead II ----
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, r_peaks_dict = nk.ecg_peaks(lead_ii, sampling_rate=fs)
        r_peaks = np.array(r_peaks_dict.get("ECG_R_Peaks", []), dtype=float)
        r_peaks = r_peaks[np.isfinite(r_peaks)].astype(int)
    except Exception:
        return features

    if r_peaks.size < 2:
        return features

    # ---- RR intervals ----
    rr = np.diff(r_peaks) / fs * 1000  # ms
    features["rr_mean_ii"] = _safe_mean(rr)
    features["rr_std_ii"]  = _safe_std(rr)
    if np.isfinite(features["rr_mean_ii"]) and features["rr_mean_ii"] > 0:
        features["heart_rate"] = 60000.0 / features["rr_mean_ii"]

    # ---- Delineation per lead ----
    def _delineate(lead_signal):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, w = nk.ecg_delineate(lead_signal, r_peaks,
                                        sampling_rate=fs, method="dwt")
            return w
        except Exception:
            return {}

    waves_ii = _delineate(lead_ii)
    waves_v1 = _delineate(lead_v1)
    waves_v2 = _delineate(lead_v2)
    waves_v3 = _delineate(lead_v3)

    # ---- ST elevation (J-point + 60 ms vs baseline) ----
    j_offset = int(0.06 * fs)

    def _st_elevation(lead_sig, waves, r_pks):
        s_offs = _get_indices(waves, "ECG_S_Peaks")
        if s_offs.size == 0:
            return np.nan
        vals = []
        for s in s_offs:
            j60 = s + j_offset
            prev = r_pks[r_pks <= s]
            if prev.size == 0:
                continue
            bl = max(0, prev[-1] - int(0.04 * fs))
            if j60 < len(lead_sig):
                vals.append(lead_sig[j60] - lead_sig[bl])
        return _safe_mean(vals) if vals else np.nan

    features["st_elevation_v1"] = _st_elevation(lead_v1, waves_v1, r_peaks)
    features["st_elevation_v2"] = _st_elevation(lead_v2, waves_v2, r_peaks)
    features["st_elevation_v3"] = _st_elevation(lead_v3, waves_v3, r_peaks)

    # ---- QRS duration (Q-onset to S-offset, ms) ----
    def _qrs_dur(waves):
        q = _get_indices(waves, "ECG_Q_Peaks")
        s = _get_indices(waves, "ECG_S_Peaks")
        n = min(len(q), len(s))
        if n == 0:
            return np.nan
        d = (s[:n] - q[:n]) / fs * 1000
        d = d[d > 0]
        return _safe_mean(d) if d.size > 0 else np.nan

    features["qrs_duration_v1"] = _qrs_dur(waves_v1)
    features["qrs_duration_v2"] = _qrs_dur(waves_v2)
    features["qrs_duration_v3"] = _qrs_dur(waves_v3)

    # ---- R-J interval on V1 (ms) ----
    s_v1 = _get_indices(waves_v1, "ECG_S_Peaks")
    if s_v1.size > 0:
        rj = []
        for s in s_v1:
            prev = r_peaks[r_peaks <= s]
            if prev.size > 0:
                v = (s - prev[-1]) / fs * 1000
                if v > 0:
                    rj.append(v)
        features["rj_interval_v1"] = _safe_mean(rj) if rj else np.nan

    # ---- P-wave duration (ms) ----
    def _p_dur(waves):
        on  = _get_indices(waves, "ECG_P_Onsets")
        off = _get_indices(waves, "ECG_P_Offsets")
        n = min(len(on), len(off))
        if n == 0:
            return np.nan
        d = (off[:n] - on[:n]) / fs * 1000
        d = d[d > 0]
        return _safe_mean(d) if d.size > 0 else np.nan

    features["p_wave_duration_ii"] = _p_dur(waves_ii)
    features["p_wave_duration_v1"] = _p_dur(waves_v1)

    # ---- P-wave amplitude (mV) ----
    def _p_amp(lead_sig, waves):
        pp = _get_indices(waves, "ECG_P_Peaks")
        pp = pp[(pp >= 0) & (pp < len(lead_sig))]
        return _safe_mean(lead_sig[pp]) if pp.size > 0 else np.nan

    features["p_wave_amplitude_ii"] = _p_amp(lead_ii, waves_ii)
    features["p_wave_amplitude_v1"] = _p_amp(lead_v1, waves_v1)

    # ---- T-wave amplitude ----
    def _t_amp(lead_sig, waves):
        tp = _get_indices(waves, "ECG_T_Peaks")
        tp = tp[(tp >= 0) & (tp < len(lead_sig))]
        return _safe_mean(lead_sig[tp]) if tp.size > 0 else np.nan

    features["t_wave_amplitude_v1"] = _t_amp(lead_v1, waves_v1)
    features["t_wave_amplitude_v2"] = _t_amp(lead_v2, waves_v2)
    features["t_wave_amplitude_v3"] = _t_amp(lead_v3, waves_v3)

    # ---- T-wave symmetry on V2 ----
    t_on  = _get_indices(waves_v2, "ECG_T_Onsets")
    t_pk  = _get_indices(waves_v2, "ECG_T_Peaks")
    t_off = _get_indices(waves_v2, "ECG_T_Offsets")
    n_s = min(len(t_on), len(t_pk), len(t_off))
    if n_s > 0:
        rising  = (t_pk[:n_s] - t_on[:n_s]).astype(float)
        falling = (t_off[:n_s] - t_pk[:n_s]).astype(float)
        ok = (rising > 0) & (falling > 0)
        if ok.any():
            features["t_wave_symmetry_v2"] = _safe_mean(rising[ok] / falling[ok])

    # ---- ST / T ratio on V2 ----
    st2 = features["st_elevation_v2"]
    t2  = features["t_wave_amplitude_v2"]
    if np.isfinite(st2) and np.isfinite(t2) and abs(t2) > 1e-6:
        features["st_t_ratio_v2"] = st2 / abs(t2)

    return features


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def build_method4_feature_dataset(
    metadata_csv_path: str = "metadata.csv",
    data_dir: str = "files",
    output_dir: str = "data_preprocessing/Steve_Method4",
    n_splits: int = 5,
    random_state: int = 42,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ---- Load ALL signals at once via the shared data_loader ----
    print("Loading raw ECG data...")
    all_signals, _ = load_raw_dataset(metadata_csv_path, data_dir)
    metadata = pd.read_csv(metadata_csv_path)
    patient_ids = metadata["patient_id"].values
    labels = (metadata["brugada"].fillna(0) > 0).astype(int).values
    n_patients = len(metadata)
    print(f"Loaded {n_patients} patients.  Signal shape: {all_signals.shape}")

    # ----     # ---- Flagged IDs (14 signal-quality flagged from Phase 1) ----
    # Source: data_preprocessing/dataset_v1_raw_drop14_dropped_ids.csv
    FLAGGED_PATIENT_IDS = {
        267630, 287355, 519139, 801261, 930107, 999474, 1088175,
        1141322, 1142915, 1180702, 1230482, 1275329, 1313974, 3058024,
    }
    flagged_ids = FLAGGED_PATIENT_IDS
    print(f"Flagged IDs: {len(flagged_ids)}")

    # ---- Filter all signals ----
    print("Applying bandpass + notch filters...")
    filtered = np.zeros_like(all_signals)
    for i in range(n_patients):
        filtered[i] = apply_clinical_filters(all_signals[i])

    # ---- Extract features per patient ----
    print("Extracting features...")
    all_feats = []
    nan_counts = {f: 0 for f in FEATURE_NAMES}

    for i in range(n_patients):
        pid = patient_ids[i]
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Patient {i+1}/{n_patients} (ID={pid})")
        try:
            feats = extract_features_single_patient(filtered[i], fs=FS)
        except Exception as e:
            print(f"  WARN: patient {pid} failed: {e}")
            feats = {name: np.nan for name in FEATURE_NAMES}

        for fn in FEATURE_NAMES:
            if not np.isfinite(feats.get(fn, np.nan)):
                nan_counts[fn] += 1

        feats["patient_id"] = pid
        feats["label"] = labels[i]
        all_feats.append(feats)

    df_all = pd.DataFrame(all_feats)[["patient_id"] + FEATURE_NAMES + ["label"]]

    print("\nNaN counts per feature:")
    for fn, cnt in nan_counts.items():
        print(f"  {fn}: {cnt}/{n_patients} ({cnt/n_patients*100:.1f}%)")

    # ---- Process both arms ----
    arms = [
        {"name": "drop14", "drop": True},
        {"name": "keepall", "drop": False},
    ]

    for arm in arms:
        arm_name = arm["name"]
        print(f"\n{'='*50}\nArm: {arm_name}\n{'='*50}")

        if arm["drop"] and flagged_ids:
            df_arm = df_all[~df_all["patient_id"].isin(flagged_ids)].copy().reset_index(drop=True)
        else:
            df_arm = df_all.copy()

        n_arm = len(df_arm)
        print(f"  Patients: {n_arm}")

        # ---- Median imputation ----
        median_fills = {}
        for fn in FEATURE_NAMES:
            col = df_arm[fn]
            finite = col[col.notna() & np.isfinite(col)]
            med = float(finite.median()) if len(finite) > 0 else 0.0
            median_fills[fn] = med
            df_arm[fn] = col.fillna(med).replace([np.inf, -np.inf], med)

        # ---- Save CSV ----
        csv_out = output_path / f"dataset_v4_features_{arm_name}.csv"
        df_arm.to_csv(csv_out, index=False)
        print(f"  Saved: {csv_out}")

        # ---- StratifiedGroupKFold ----
        y = df_arm["label"].values
        pids = df_arm["patient_id"].values
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                     random_state=random_state)

        fold_meta = {
            "n_splits": n_splits, "random_state": random_state,
            "arm_name": arm_name, "n_patients": n_arm,
            "feature_names": FEATURE_NAMES, "folds": {},
        }
        for fi, (tr, te) in enumerate(sgkf.split(df_arm[FEATURE_NAMES].values, y, groups=pids)):
            fold_meta["folds"][str(fi)] = {
                "train": {"patient_ids": sorted(pids[tr].tolist()),
                          "indices": tr.astype(int).tolist()},
                "test":  {"patient_ids": sorted(pids[te].tolist()),
                          "indices": te.astype(int).tolist()},
            }
        json_out = output_path / f"fold_composition_v4_{arm_name}.json"
        with open(json_out, "w") as f:
            json.dump(fold_meta, f, indent=2)
        print(f"  Saved: {json_out}")

        # ---- Report ----
        report = {
            "arm_name": arm_name,
            "total_patients": n_arm,
            "n_features": len(FEATURE_NAMES),
            "nan_counts_before_imputation": {k: int(v) for k, v in nan_counts.items()},
            "median_fill_values": median_fills,
            "class_distribution": {"positive": int(y.sum()), "negative": int((y == 0).sum())},
            "fold_sizes": {str(i): {"train": len(fold_meta["folds"][str(i)]["train"]["indices"]),
                                     "test": len(fold_meta["folds"][str(i)]["test"]["indices"])}
                           for i in range(n_splits)},
        }
        rpt_out = output_path / f"method4_extraction_report_{arm_name}.json"
        with open(rpt_out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  Saved: {rpt_out}")

    print("\n✅ Method 4 complete.")


# ---------------------------------------------------------------------------
# Fold loader (for downstream model training)
# ---------------------------------------------------------------------------
def load_feature_dataset_for_fold(
    fold_id: int,
    arm_name: str = "keepall",
    data_dir: str = "data_preprocessing/Steve_Method4",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load fold with StandardScaler (fit train, transform test)."""
    dp = Path(data_dir)
    df = pd.read_csv(dp / f"dataset_v4_features_{arm_name}.csv")
    with open(dp / f"fold_composition_v4_{arm_name}.json") as f:
        meta = json.load(f)

    fk = str(fold_id)
    tr = np.array(meta["folds"][fk]["train"]["indices"], dtype=int)
    te = np.array(meta["folds"][fk]["test"]["indices"], dtype=int)
    fcols = meta.get("feature_names", FEATURE_NAMES)

    X_tr = df.iloc[tr][fcols].values.astype(np.float32)
    X_te = df.iloc[te][fcols].values.astype(np.float32)
    y_tr = df.iloc[tr]["label"].values.astype(int)
    y_te = df.iloc[te]["label"].values.astype(int)

    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr).astype(np.float32)
    X_te = sc.transform(X_te).astype(np.float32)
    return X_tr, X_te, y_tr, y_te


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Method 4: ECG Feature Engineering for Brugada Classification"
    )
    parser.add_argument("metadata_csv_path", type=str)
    parser.add_argument("data_dir", type=str, help="Directory with WFDB patient folders (e.g. 'files')")
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    build_method4_feature_dataset(
        args.metadata_csv_path, args.data_dir, args.output_dir,
        args.n_splits, args.random_state,
    )
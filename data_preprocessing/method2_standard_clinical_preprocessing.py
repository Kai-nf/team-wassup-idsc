import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.preprocessing import StandardScaler

# Add repo root to Python path so we can import the data loader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Environment_setup.data_loader import load_raw_dataset

# =============================================================================
# HELPERS
# =============================================================================

def _binarize_brugada_labels(labels):
    """Positive if brugada > 0, matches Method 3 labelling convention."""
    labels_numeric = pd.to_numeric(pd.Series(labels), errors='coerce')
    return (labels_numeric.fillna(0) > 0).astype(int)

def apply_clinical_filters(signal, fs=100):
    """
    Apply Bandpass (0.5-40 Hz, order 3) and Notch (50 Hz) filters.
    signal shape: (T, 12) -> (samples, leads)
    """
    # 1. Bandpass 0.5-40 Hz
    nyquist = 0.5 * fs
    b, a = butter(3, [0.5 / nyquist, 40.0 / nyquist], btype='band')
    filtered = filtfilt(b, a, signal, axis=0)

    # 2. Notch 50 Hz — safe implementation avoiding Nyquist boundary crash
    b_n, a_n = iirnotch(50.0, 30.0, fs=fs)
    filtered = filtfilt(b_n, a_n, filtered, axis=0)

    return filtered

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_preprocessing_pipeline(
    all_signals,
    labels,
    metadata_csv_path,
    dropped_csv_path,
    manifest_json_path,
    output_dir,
    arm_name='drop14',
    n_splits=5,
):
    """
    Method 2 — Standard Clinical Preprocessing pipeline (JSON-Enforced Folds).
    """
    # ── 1. Load metadata and validate alignment ────────────────────────────────
    metadata   = pd.read_csv(metadata_csv_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    labels_arr = _binarize_brugada_labels(labels).to_numpy()
    if len(metadata) != len(all_signals) or len(labels_arr) != len(all_signals):
        raise ValueError("Input length mismatch: metadata, labels, and signals must align.")

    metadata_ids = metadata['patient_id'].astype(str)

    # ── 2. Apply Method 1 dropped IDs ────────────────────────────────────────
    dropped_ids  = set()
    dropped_file = Path(dropped_csv_path)
    if dropped_file.exists() and dropped_file.stat().st_size > 0:
        dropped_df = pd.read_csv(dropped_file)
        pid_col    = next((col for col in dropped_df.columns if 'patient_id' in str(col).strip().lower()), None)
        if pid_col is not None:
            dropped_ids = set(dropped_df[pid_col].dropna().astype(str).tolist())

    keep_mask = ~metadata_ids.isin(dropped_ids).to_numpy()
    if keep_mask.sum() == 0:
        raise ValueError("No records left after applying Method 1 dropped IDs.")

    X_clean              = all_signals[keep_mask]
    y_clean              = labels_arr[keep_mask]
    ids_clean            = metadata_ids.to_numpy()[keep_mask]
    
    print(f"Cohort: retained {keep_mask.sum()} | excluded {(~keep_mask).sum()}")

    # ── 3. Filter ALL retained signals once ────────────────────────────────────
    print(f"Applying clinical filters to {len(X_clean)} signals once...")
    X_filtered = np.array(
        [apply_clinical_filters(X_clean[i]) for i in range(len(X_clean))],
        dtype=np.float32,
    )
    print("Filtering complete.")

    # ── 4. Build fold splits strictly from JSON Master Manifest ────────────────
    manifest_file = Path(manifest_json_path)
    if not manifest_file.exists():
        raise FileNotFoundError(f"[ABORT] Cannot find {manifest_file}. All models MUST use the same manifest.")
        
    with open(manifest_file, 'r') as f:
        master_folds = json.load(f)

    # Map the raw string patient IDs from the JSON to the integer indices of X_clean
    pid_to_clean_idx = {str(pid): idx for idx, pid in enumerate(ids_clean)}

    fold_indices = []
    for fold_idx in range(n_splits):
        fold_key = str(fold_idx)
        train_pids = master_folds["folds"][fold_key]["train"]["patient_ids"]
        test_pids = master_folds["folds"][fold_key]["test"]["patient_ids"]
        
        # Convert string IDs to integer array indices (only if they survived the drop mask)
        train_idx = np.array([pid_to_clean_idx[str(pid)] for pid in train_pids if str(pid) in pid_to_clean_idx], dtype=int)
        test_idx = np.array([pid_to_clean_idx[str(pid)] for pid in test_pids if str(pid) in pid_to_clean_idx], dtype=int)
        
        fold_indices.append((train_idx, test_idx))
        
    fold_source = manifest_file.name
    print(f"Fold splits strictly loaded from {fold_source}")

    # ── 5. Per-fold scaling (fit on train only) ───────────────────────────────
    folds = []
    for fold_idx, (train_idx, test_idx) in enumerate(fold_indices):
        X_train_filt = X_filtered[train_idx]
        X_test_filt  = X_filtered[test_idx]

        N_train, T, L = X_train_filt.shape
        N_test        = X_test_filt.shape[0]

        scaler         = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_filt.reshape(-1, L)).reshape(N_train, T, L).astype(np.float32)
        X_test_scaled  = scaler.transform(X_test_filt.reshape(-1, L)).reshape(N_test, T, L).astype(np.float32)

        n_pos = int(y_clean[test_idx].sum())
        print(f"  Fold {fold_idx}: train={N_train}  test={N_test} (pos={n_pos})")

        folds.append({
            'fold':     fold_idx,
            'X_train':  X_train_scaled,
            'X_test':   X_test_scaled,
            'y_train':  y_clean[train_idx].astype(np.int8),
            'y_test':   y_clean[test_idx].astype(np.int8),
            'id_train': ids_clean[train_idx],
            'id_test':  ids_clean[test_idx],
        })

    # ── 6. Save ───────────────────────────────────────────────────────────────
    save_path = output_path / f'dataset_v2_filtered_{arm_name}.npy'
    np.save(
        str(save_path),
        {
            'folds':            folds,
            'n_splits':         n_splits,
            'arm_name':         arm_name,
            'retained_records': int(keep_mask.sum()),
            'excluded_records': int((~keep_mask).sum()),
            'fold_source':      fold_source,
        },
    )

    print(f"\nMethod 2 ({arm_name}) complete. Saved to: {save_path}")
    return folds

# =============================================================================
# EXECUTION
# =============================================================================
if __name__ == '__main__':
    # Dynamically find the root of your repository
    REPO_ROOT = Path(__file__).resolve().parent.parent

    # New Paths based on your cleanup
    CSV_PATH      = REPO_ROOT / 'Environment_setup' / 'metadata.csv'
    DATA_DIR      = REPO_ROOT / 'Environment_setup' / 'files'
    MANIFEST_PATH = REPO_ROOT / 'master_folds_drop14.json'
    OUTPUT_DIR    = REPO_ROOT / 'Preprocessed_Dataset'
    arm_name      = 'drop14'
    DROPPED_CSV   = REPO_ROOT / 'felicia' / 'data_preprocessing_method1' / f'dataset_v1_raw_{arm_name}_dropped_ids.csv'

    print('Loading raw dataset...')
    all_signals, labels = load_raw_dataset(str(CSV_PATH), str(DATA_DIR))
    print(f'Loaded {all_signals.shape[0]} recordings.')

    print(f"\nRunning Method 2 standard clinical preprocessing (arm='{arm_name}')...")
    folds = run_preprocessing_pipeline(
        all_signals=all_signals,
        labels=labels,
        metadata_csv_path=str(CSV_PATH),
        dropped_csv_path=str(DROPPED_CSV),
        manifest_json_path=str(MANIFEST_PATH),
        output_dir=str(OUTPUT_DIR),
        arm_name=arm_name,
    )
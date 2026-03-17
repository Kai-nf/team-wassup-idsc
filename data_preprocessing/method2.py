import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

def _binarize_brugada_labels(labels):
    # Match Method 3 labeling: positive if brugada > 0, otherwise negative.
    labels_numeric = pd.to_numeric(pd.Series(labels), errors='coerce')
    return (labels_numeric.fillna(0) > 0).astype(int)

def apply_clinical_filters(signal, fs=100):
    """
    Applies Bandpass (0.5-40Hz) and Notch (50Hz) filters.
    signal shape: (1200, 12) -> (samples, leads)
    """
    # 1. Bandpass Filter: 0.5–40 Hz
    nyquist = 0.5 * fs
    low, high = 0.5 / nyquist, 40 / nyquist
    b, a = butter(3, [low, high], btype='band')
    # apply along axis 0 (time axis)
    filtered_signal = filtfilt(b, a, signal, axis=0)
    
    # 2. Notch Filter: 50 Hz
    b_notch, a_notch = iirnotch(50, 30, fs)
    filtered_signal = filtfilt(b_notch, a_notch, filtered_signal, axis=0)
    
    return filtered_signal

def run_preprocessing_pipeline(
    all_signals,
    labels,
    metadata_csv_path='metadata.csv',
    dropped_csv_path='felicia/data_preprocessing_method1/dataset_v1_raw_drop14_dropped_ids.csv',
    manifest_csv_path='felicia/data_preprocessing_method1/dataset_v1_raw_drop14_manifest.csv',
    output_dir='data_preprocessing',
    arm_name='drop14',
    n_splits=5,
    random_state=42,
):
    """
    Method 2 (Standard Clinical Preprocessing):
    - Bandpass filter 0.5-40 Hz + Notch 50 Hz
    - StandardScaler per-lead, fit on train only per fold
    - Binarize labels as 1 if brugada > 0 else 0 (matches Method 3)
    - Uses Method 1 dropped IDs for standardized cohort
    - Reuses Method 1 fold assignments when manifest is available
    - Fallback split uses StratifiedGroupKFold grouped by patient_id
    - Save as arm-specific dataset_v2_filtered_{arm}.npy
    """
    metadata = pd.read_csv(metadata_csv_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    labels_arr = _binarize_brugada_labels(labels).to_numpy()
    if len(metadata) != len(all_signals) or len(labels_arr) != len(all_signals):
        raise ValueError(
            "Input length mismatch: metadata, labels, and signals must align by row order."
        )

    metadata_ids = metadata['patient_id'].astype(str)

    dropped_ids = set()
    dropped_file = Path(dropped_csv_path)
    if dropped_file.exists() and dropped_file.stat().st_size > 0:
        dropped_df = pd.read_csv(dropped_file)
        pid_col = next(
            (col for col in dropped_df.columns if 'patient_id' in str(col).strip().lower()),
            None,
        )
        if pid_col is not None:
            dropped_ids = set(dropped_df[pid_col].dropna().astype(str).tolist())

    keep_mask = ~metadata_ids.isin(dropped_ids).to_numpy()
    if keep_mask.sum() == 0:
        raise ValueError("No records left after applying Method 1 dropped IDs.")

    X_clean = all_signals[keep_mask]
    y_clean = labels_arr[keep_mask]
    ids_clean = metadata_ids.to_numpy()[keep_mask]
    clean_original_indices = np.where(keep_mask)[0]

    fold_indices = []
    fold_source = 'stratified_group_kfold'
    manifest_file = Path(manifest_csv_path)
    if manifest_file.exists() and manifest_file.stat().st_size > 0:
        manifest_df = pd.read_csv(manifest_file)
        if len(manifest_df) == len(metadata) and 'test_fold' in manifest_df.columns:
            test_fold_series = pd.to_numeric(manifest_df['test_fold'], errors='coerce').fillna(-1).astype(int)
            orig_to_clean = {orig_idx: clean_idx for clean_idx, orig_idx in enumerate(clean_original_indices)}

            reusable = True
            for fold_idx in range(n_splits):
                fold_test_orig = np.where((test_fold_series.to_numpy() == fold_idx) & keep_mask)[0]
                if len(fold_test_orig) == 0:
                    reusable = False
                    break

                test_idx = np.array([orig_to_clean[i] for i in fold_test_orig], dtype=int)
                test_set = set(test_idx.tolist())
                train_idx = np.array(
                    [idx for idx in range(len(X_clean)) if idx not in test_set],
                    dtype=int,
                )
                fold_indices.append((train_idx, test_idx))

            if reusable and len(fold_indices) == n_splits:
                fold_source = 'method1_manifest'
            else:
                fold_indices = []

    if not fold_indices:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        fold_indices = list(sgkf.split(X_clean, y_clean, groups=ids_clean))

    folds = []

    for fold_idx, (train_idx, test_idx) in enumerate(fold_indices):
        X_train_filt = np.array([apply_clinical_filters(X_clean[i]) for i in train_idx])
        X_test_filt = np.array([apply_clinical_filters(X_clean[i]) for i in test_idx])

        N_train, T, L = X_train_filt.shape
        N_test = X_test_filt.shape[0]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_filt.reshape(-1, L)).reshape(N_train, T, L)
        X_test_scaled  = scaler.transform(X_test_filt.reshape(-1, L)).reshape(N_test, T, L)

        folds.append({
            'fold': fold_idx,
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_clean[train_idx],
            'y_test': y_clean[test_idx],
            'id_train': ids_clean[train_idx],
            'id_test': ids_clean[test_idx],
        })

    np.save(
        output_path / f'dataset_v2_filtered_{arm_name}.npy',
        {
            'folds': folds,
            'n_splits': n_splits,
            'arm_name': arm_name,
            'retained_records': int(keep_mask.sum()),
            'excluded_records': int((~keep_mask).sum()),
            'fold_source': fold_source,
            'random_state': random_state,
        },
    )
    print(f"Method 2 ({arm_name}) complete. Saved dataset_v2_filtered_{arm_name}.npy")
    print(
        f"Method 2 cohort: retained {int(keep_mask.sum())} | excluded {int((~keep_mask).sum())}"
        f" | folds from {fold_source}"
    )

    return folds
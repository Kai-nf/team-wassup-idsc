import numpy as np
import pandas as pd
from pathlib import Path
import json
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def _append_reason(existing, reason):
    if not existing:
        return reason
    return f"{existing};{reason}"


def run_raw_baseline_pipeline(
    all_signals,
    labels,
    metadata_csv_path='metadata.csv',
    flagged_csv_path='data_preprocessing/flagged_recordings_phase1.csv',
    output_dir='data_preprocessing',
    arm_name='drop14',
    drop_flagged_in_phase1=True,
    n_splits=5,
    random_state=42,
):
    """
    Method 1 (Raw Baseline):
    - Keep raw signals (no filtering)
    - Optional drop of recordings flagged during Phase 1 EDA
    - 5-Fold Stratified K-Fold Cross-Validation
    - Save dataset and audit artifacts
    """
    metadata = pd.read_csv(metadata_csv_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if len(metadata) != len(all_signals) or len(labels) != len(all_signals):
        raise ValueError(
            "Input length mismatch: metadata, labels, and signals must align by row order."
        )

    metadata_ids = metadata['patient_id'].astype(str)
    labels_series = pd.Series(labels)

    flagged_ids = set()
    flagged_file = Path(flagged_csv_path)
    if flagged_file.exists() and flagged_file.stat().st_size > 0:
        flagged_df = pd.read_csv(flagged_file)
        pid_col = next(
            (col for col in flagged_df.columns if 'patient_id' in str(col).strip().lower()),
            None,
        )
        if pid_col is not None:
            flagged_ids = set(flagged_df[pid_col].dropna().astype(str).tolist())

    flagged_mask = metadata_ids.isin(flagged_ids)
    if not drop_flagged_in_phase1:
        flagged_mask = pd.Series(False, index=metadata.index)

    finite_mask = np.isfinite(all_signals).all(axis=(1, 2))
    missing_label_mask = labels_series.isna().to_numpy()

    keep_mask = (~flagged_mask.to_numpy()) & finite_mask & (~missing_label_mask)

    dropped_reason = pd.Series('', index=metadata.index, dtype='string')
    for idx in metadata.index[flagged_mask]:
        dropped_reason.iloc[idx] = _append_reason(dropped_reason.iloc[idx], 'flagged_in_phase1')
    for idx in metadata.index[~finite_mask]:
        dropped_reason.iloc[idx] = _append_reason(dropped_reason.iloc[idx], 'non_finite_signal')
    for idx in metadata.index[missing_label_mask]:
        dropped_reason.iloc[idx] = _append_reason(dropped_reason.iloc[idx], 'missing_label')

    X_clean = all_signals[keep_mask]
    y_clean = labels_series.to_numpy()[keep_mask]
    ids_clean = metadata_ids.to_numpy()[keep_mask]

    if len(X_clean) == 0:
        raise ValueError("No records left after exclusion. Check flagged list and source data.")

    class_counts = pd.Series(y_clean).value_counts().sort_index()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = []
    clean_original_indices = np.where(keep_mask)[0]
    test_fold_assignments = np.full(len(X_clean), -1, dtype=int)

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_clean, y_clean)):
        folds.append({
            'fold': fold_idx,
            'X_train': X_clean[train_idx],
            'X_test': X_clean[test_idx],
            'y_train': y_clean[train_idx],
            'y_test': y_clean[test_idx],
            'id_train': ids_clean[train_idx],
            'id_test': ids_clean[test_idx],
        })
        test_fold_assignments[test_idx] = fold_idx

    dataset_v1 = {
        'folds': folds,
        'n_splits': n_splits,
        'random_state': random_state,
        'arm_name': arm_name,
        'exclusion_policy': [
            'flagged_in_phase1',
            'non_finite_signal',
            'missing_label',
        ],
    }
    np.save(output_path / f'dataset_v1_raw_{arm_name}.npy', dataset_v1)

    test_fold_col = pd.Series(-1, index=metadata.index, dtype=int)
    for i, orig_idx in enumerate(clean_original_indices):
        test_fold_col.iloc[orig_idx] = test_fold_assignments[i]

    manifest = pd.DataFrame(
        {
            'patient_id': metadata_ids,
            'label': labels_series,
            'included': keep_mask,
            'drop_reason': dropped_reason,
            'test_fold': test_fold_col,
        }
    )
    manifest.to_csv(output_path / f'dataset_v1_raw_{arm_name}_manifest.csv', index=False)
    manifest.loc[~manifest['included']].to_csv(
        output_path / f'dataset_v1_raw_{arm_name}_dropped_ids.csv', index=False
    )

    fold_sizes = [
        {'fold': f['fold'], 'train': len(f['X_train']), 'test': len(f['X_test'])}
        for f in folds
    ]
    summary = {
        'total_records': int(len(metadata)),
        'excluded_records': int((~keep_mask).sum()),
        'retained_records': int(keep_mask.sum()),
        'arm_name': arm_name,
        'n_splits': n_splits,
        'fold_sizes': fold_sizes,
        'class_counts_after_exclusion': {
            str(k): int(v) for k, v in class_counts.to_dict().items()
        },
        'excluded_by_reason': {
            'flagged_in_phase1': int(flagged_mask.sum()),
            'non_finite_signal': int((~finite_mask).sum()),
            'missing_label': int(missing_label_mask.sum()),
        },
    }
    with open(output_path / f'dataset_v1_raw_{arm_name}_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"Method 1 ({arm_name}) complete. Saved dataset_v1_raw_{arm_name}.npy and audit artifacts.")
    print(f"Retained records: {summary['retained_records']} | Excluded: {summary['excluded_records']}")

    return folds

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
    dropped_csv_path='data_preprocessing/dataset_v1_raw_drop14_dropped_ids.csv',
    manifest_csv_path='data_preprocessing/dataset_v1_raw_drop14_manifest.csv',
    output_dir='data_preprocessing',
    arm_name='drop14',
    n_splits=5,
    random_state=42,
):
    """
    Method 2 (Standard Clinical Preprocessing):
    - Bandpass filter 0.5-40 Hz + Notch 50 Hz
    - StandardScaler per-lead, fit on train only per fold
    - Uses Method 1 dropped IDs for standardized cohort
    - Reuses Method 1 fold assignments when manifest is available
    - Save as arm-specific dataset_v2_filtered_{arm}.npy
    """
    metadata = pd.read_csv(metadata_csv_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    labels_arr = np.asarray(labels)
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
    fold_source = 'stratified_kfold'
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
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        fold_indices = list(skf.split(X_clean, y_clean))

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
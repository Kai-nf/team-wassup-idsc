import numpy as np
import pandas as pd
from pathlib import Path
import json
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.model_selection import train_test_split
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
    test_size=0.30,
    random_state=42,
):
    """
    Method 1 (Raw Baseline):
    - Keep raw signals (no filtering)
    - Drop recordings flagged during Phase 1 EDA
    - Holdout split 70/30 stratified by class
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
        if 'patient_id' in flagged_df.columns:
            flagged_ids = set(flagged_df['patient_id'].dropna().astype(str).tolist())

    flagged_mask = metadata_ids.isin(flagged_ids)
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
    try:
        X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
            X_clean,
            y_clean,
            ids_clean,
            test_size=test_size,
            random_state=random_state,
            stratify=y_clean,
        )
    except ValueError as exc:
        raise ValueError(
            f"Stratified split failed. Class counts after exclusion: {class_counts.to_dict()}"
        ) from exc

    dataset_v1 = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'id_train': id_train,
        'id_test': id_test,
        'split_ratio': {'train': 1 - test_size, 'test': test_size},
        'random_state': random_state,
        'exclusion_policy': [
            'flagged_in_phase1',
            'non_finite_signal',
            'missing_label',
        ],
    }
    np.save(output_path / 'dataset_v1_raw.npy', dataset_v1)

    manifest = pd.DataFrame(
        {
            'patient_id': metadata_ids,
            'label': labels_series,
            'included': keep_mask,
            'drop_reason': dropped_reason,
            'split': '',
        }
    )

    train_set = set(id_train.tolist())
    test_set = set(id_test.tolist())
    manifest.loc[manifest['included'] & manifest['patient_id'].isin(train_set), 'split'] = 'train'
    manifest.loc[manifest['included'] & manifest['patient_id'].isin(test_set), 'split'] = 'test'

    manifest.to_csv(output_path / 'dataset_v1_raw_manifest.csv', index=False)
    manifest.loc[~manifest['included']].to_csv(
        output_path / 'dataset_v1_raw_dropped_ids.csv', index=False
    )

    summary = {
        'total_records': int(len(metadata)),
        'excluded_records': int((~keep_mask).sum()),
        'retained_records': int(keep_mask.sum()),
        'train_records': int(len(X_train)),
        'test_records': int(len(X_test)),
        'class_counts_after_exclusion': {
            str(k): int(v) for k, v in class_counts.to_dict().items()
        },
        'excluded_by_reason': {
            'flagged_in_phase1': int(flagged_mask.sum()),
            'non_finite_signal': int((~finite_mask).sum()),
            'missing_label': int(missing_label_mask.sum()),
        },
    }
    with open(output_path / 'dataset_v1_raw_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print("Method 1 complete. Saved dataset_v1_raw.npy and audit artifacts.")
    print(f"Retained records: {summary['retained_records']} | Excluded: {summary['excluded_records']}")

    return X_train, X_test

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

def run_preprocessing_pipeline(all_signals, labels):
    """
    Handles Splitting, Filtering, and Scaling.
    """
    # 3. Holdout Split: 70% train / 30% test, stratified by class
    X_train, X_test, y_train, y_test = train_test_split(
        all_signals, 
        labels, 
        test_size=0.30, 
        random_state=42, 
        stratify=labels
    )

    # 4. Apply Filters to all splits
    # We apply filters individually to each 12-lead recording
    X_train_filt = np.array([apply_clinical_filters(record) for record in X_train])
    X_test_filt = np.array([apply_clinical_filters(record) for record in X_test])

    # 5. Feature Scaling: StandardScaler applied per-lead
    # StandardScaler expects (n_samples, n_features). 
    # We treat each (timestep x lead) as a feature or flatten.
    # Standard practice: Scale across all timesteps for each lead.
    
    scaler = StandardScaler()
    
    # Reshape to (N * 1200, 12) to fit scaler on lead-columns
    N_train, T, L = X_train_filt.shape
    N_test = X_test_filt.shape[0]
    
    X_train_reshaped = X_train_filt.reshape(-1, L)
    X_test_reshaped = X_test_filt.reshape(-1, L)

    # Fit ON TRAIN ONLY, transform on both
    X_train_scaled = scaler.fit_transform(X_train_reshaped)
    X_test_scaled = scaler.transform(X_test_reshaped)

    # Reshape back to original 3D format (N, 1200, 12)
    X_train_final = X_train_scaled.reshape(N_train, T, L)
    X_test_final = X_test_scaled.reshape(N_test, T, L)

    # 6. Save as dataset_v2_filtered.npy
    output_data = {
        'X_train': X_train_final,
        'X_test': X_test_final,
        'y_train': y_train,
        'y_test': y_test
    }
    np.save('data_preprocessing/dataset_v2_filtered.npy', output_data)
    print("Preprocessing complete. File saved as dataset_v2_filtered.npy")

    return X_train_final, X_test_final
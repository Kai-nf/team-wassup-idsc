import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def _append_reason(existing, reason):
    if not existing:
        return reason
    return f"{existing};{reason}"


def _binarize_brugada_labels(labels):
    # Match Method 3 labeling: positive if brugada > 0, otherwise negative.
    labels_numeric = pd.to_numeric(pd.Series(labels), errors='coerce')
    return (labels_numeric.fillna(0) > 0).astype(int)


def run_raw_baseline_pipeline(
    all_signals,
    labels,
    metadata_csv_path='metadata.csv',
    flagged_csv_path='flagged_recordings_phase1.csv',
    output_dir='felicia/data_preprocessing_method1',
    arm_name='drop14',
    drop_flagged_in_phase1=True,
    n_splits=5,
    random_state=42,
):
    """
    Method 1 (Raw Baseline):
    - Keep raw signals (no filtering)
    - Binarize labels as 1 if brugada > 0 else 0 (matches Method 3)
    - Optional drop of recordings flagged during Phase 1 EDA
    - 5-Fold Stratified Group K-Fold Cross-Validation (grouped by patient_id)
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
    labels_series = _binarize_brugada_labels(labels)

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
    keep_mask = (~flagged_mask.to_numpy()) & finite_mask

    dropped_reason = pd.Series('', index=metadata.index, dtype='string')
    for idx in metadata.index[flagged_mask]:
        dropped_reason.iloc[idx] = _append_reason(dropped_reason.iloc[idx], 'flagged_in_phase1')
    for idx in metadata.index[~finite_mask]:
        dropped_reason.iloc[idx] = _append_reason(dropped_reason.iloc[idx], 'non_finite_signal')

    X_clean = all_signals[keep_mask]
    y_clean = labels_series.to_numpy()[keep_mask]
    ids_clean = metadata_ids.to_numpy()[keep_mask]

    if len(X_clean) == 0:
        raise ValueError("No records left after exclusion. Check flagged list and source data.")

    class_counts = pd.Series(y_clean).value_counts().sort_index()

    groups_clean = ids_clean
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = []
    clean_original_indices = np.where(keep_mask)[0]
    test_fold_assignments = np.full(len(X_clean), -1, dtype=int)

    for fold_idx, (train_idx, test_idx) in enumerate(
        sgkf.split(X_clean, y_clean, groups=groups_clean)
    ):
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
        },
    }
    with open(output_path / f'dataset_v1_raw_{arm_name}_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"Method 1 ({arm_name}) complete. Saved dataset_v1_raw_{arm_name}.npy and audit artifacts.")
    print(f"Retained records: {summary['retained_records']} | Excluded: {summary['excluded_records']}")

    return folds
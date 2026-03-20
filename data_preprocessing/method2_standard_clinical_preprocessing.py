import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


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

    Parameters
    ----------
    signal : np.ndarray  shape (T, 12) — (samples, leads)
    fs     : int         sampling frequency in Hz

    Returns
    -------
    np.ndarray  shape (T, 12), filtered along axis=0 (time axis)
    """
    # 1. Bandpass 0.5-40 Hz
    nyquist = 0.5 * fs
    b, a = butter(3, [0.5 / nyquist, 40.0 / nyquist], btype='band')
    filtered = filtfilt(b, a, signal, axis=0)

    # 2. Notch 50 Hz — iirnotch avoids the Nyquist boundary issue of
    #    butter bandstop at fs=100 Hz (50 Hz / Nyquist = 1.0 exactly)
    b_n, a_n = iirnotch(50.0, 30.0, fs=fs)
    filtered = filtfilt(b_n, a_n, filtered, axis=0)

    return filtered


# =============================================================================
# MAIN PIPELINE
# =============================================================================

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
    Method 2 — Standard Clinical Preprocessing pipeline.

    Steps
    -----
    1. Load metadata; align signals, labels, and patient IDs by row order.
    2. Drop patients listed in Method 1's dropped-IDs CSV (standardised cohort).
    3. Filter ALL remaining signals once (bandpass + notch) — not per fold.
    4. Build fold splits:
         a. Reuse Method 1 manifest when available (preferred — same patient splits).
         b. Fall back to StratifiedGroupKFold grouped by patient_id.
    5. Per fold: fit StandardScaler on training signals only, transform test.
    6. Save output as dataset_v2_filtered_{arm_name}.npy (pickled dict).

    Parameters
    ----------
    all_signals          : np.ndarray  (N, T, 12)
    labels               : array-like  length N, raw Brugada labels
    metadata_csv_path    : str path to metadata CSV (must have 'patient_id' column)
    dropped_csv_path     : str path to Method 1 dropped-IDs CSV
    manifest_csv_path    : str path to Method 1 fold-assignment manifest CSV
    output_dir           : str directory where the .npy file is saved
    arm_name             : str appended to the output filename, e.g. 'drop14'
    n_splits             : int number of cross-validation folds
    random_state         : int random seed for fallback StratifiedGroupKFold

    Returns
    -------
    folds : list of dicts, one per fold (see fold dict schema below)

    Output .npy fold dict schema
    ----------------------------
    Each fold dict contains:
        fold          : int
        X_train       : np.ndarray (N_train, T, 12)  scaled
        X_test        : np.ndarray (N_test,  T, 12)  scaled
        y_train       : np.ndarray (N_train,)         int8
        y_test        : np.ndarray (N_test,)           int8
        id_train      : np.ndarray (N_train,)          str  patient IDs
        id_test       : np.ndarray (N_test,)            str  patient IDs

    Load the saved file with:
        data = np.load(path, allow_pickle=True).item()
        folds = data['folds']
    """
    # ── 1. Load metadata and validate alignment ────────────────────────────────
    metadata   = pd.read_csv(metadata_csv_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    labels_arr = _binarize_brugada_labels(labels).to_numpy()
    if len(metadata) != len(all_signals) or len(labels_arr) != len(all_signals):
        raise ValueError(
            "Input length mismatch: metadata, labels, and signals must align by row order.\n"
            f"  metadata rows : {len(metadata)}\n"
            f"  signals rows  : {len(all_signals)}\n"
            f"  labels rows   : {len(labels_arr)}"
        )

    metadata_ids = metadata['patient_id'].astype(str)

    # ── 2. Apply Method 1 dropped IDs ────────────────────────────────────────
    dropped_ids  = set()
    dropped_file = Path(dropped_csv_path)
    if dropped_file.exists() and dropped_file.stat().st_size > 0:
        dropped_df = pd.read_csv(dropped_file)
        pid_col    = next(
            (col for col in dropped_df.columns if 'patient_id' in str(col).strip().lower()),
            None,
        )
        if pid_col is not None:
            dropped_ids = set(dropped_df[pid_col].dropna().astype(str).tolist())

    keep_mask = ~metadata_ids.isin(dropped_ids).to_numpy()
    if keep_mask.sum() == 0:
        raise ValueError("No records left after applying Method 1 dropped IDs.")

    X_clean              = all_signals[keep_mask]
    y_clean              = labels_arr[keep_mask]
    ids_clean            = metadata_ids.to_numpy()[keep_mask]
    clean_original_indices = np.where(keep_mask)[0]

    print(f"Cohort: retained {keep_mask.sum()} | excluded {(~keep_mask).sum()}")

    # ── 3. Filter ALL retained signals once (FIX: was per-fold = 5× redundant) ─
    print(f"Applying clinical filters to {len(X_clean)} signals once...")
    X_filtered = np.array(
        [apply_clinical_filters(X_clean[i]) for i in range(len(X_clean))],
        dtype=np.float32,
    )
    print("Filtering complete.")

    # ── 4. Build fold splits ──────────────────────────────────────────────────
    fold_indices = []
    fold_source  = 'stratified_group_kfold'

    manifest_file = Path(manifest_csv_path)
    if manifest_file.exists() and manifest_file.stat().st_size > 0:
        manifest_df = pd.read_csv(manifest_file)
        if len(manifest_df) == len(metadata) and 'test_fold' in manifest_df.columns:
            test_fold_series = (
                pd.to_numeric(manifest_df['test_fold'], errors='coerce')
                .fillna(-1)
                .astype(int)
            )
            # Map original row indices -> cleaned (post-drop) row indices
            orig_to_clean = {
                int(orig_idx): clean_idx
                for clean_idx, orig_idx in enumerate(clean_original_indices)
            }

            reusable = True
            for fold_idx in range(n_splits):
                fold_test_orig = np.where(
                    (test_fold_series.to_numpy() == fold_idx) & keep_mask
                )[0]
                if len(fold_test_orig) == 0:
                    reusable = False
                    break

                test_idx  = np.array([orig_to_clean[i] for i in fold_test_orig], dtype=int)
                test_set  = set(test_idx.tolist())
                train_idx = np.array(
                    [idx for idx in range(len(X_clean)) if idx not in test_set],
                    dtype=int,
                )
                fold_indices.append((train_idx, test_idx))

            if reusable and len(fold_indices) == n_splits:
                fold_source = 'method1_manifest'
                print(f"Fold splits loaded from Method 1 manifest.")
            else:
                fold_indices = []
                print("Method 1 manifest not fully reusable — falling back to StratifiedGroupKFold.")

    if not fold_indices:
        print("Building folds with StratifiedGroupKFold...")
        sgkf         = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        fold_indices = list(sgkf.split(X_filtered, y_clean, groups=ids_clean))

    # ── 5. Per-fold scaling (fit on train only) ───────────────────────────────
    folds = []

    for fold_idx, (train_idx, test_idx) in enumerate(fold_indices):
        X_train_filt = X_filtered[train_idx]   # already filtered in step 3
        X_test_filt  = X_filtered[test_idx]

        N_train, T, L = X_train_filt.shape
        N_test        = X_test_filt.shape[0]

        # StandardScaler: fit on train leads only, transform test
        scaler        = StandardScaler()
        X_train_scaled = scaler.fit_transform(
            X_train_filt.reshape(-1, L)
        ).reshape(N_train, T, L).astype(np.float32)

        X_test_scaled  = scaler.transform(
            X_test_filt.reshape(-1, L)
        ).reshape(N_test, T, L).astype(np.float32)

        n_pos = int(y_clean[test_idx].sum())
        n_neg = N_test - n_pos
        print(f"  Fold {fold_idx}: train={N_train}  test={N_test} (pos={n_pos}, neg={n_neg})")

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
    # Saved as a pickled dict — load with np.load(path, allow_pickle=True).item()
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
            'random_state':     random_state,
        },
    )

    print(f"\nMethod 2 ({arm_name}) complete.")
    print(f"  Output : {save_path}")
    print(f"  Cohort : {int(keep_mask.sum())} retained, {int((~keep_mask).sum())} excluded")
    print(f"  Folds  : {fold_source}")
    print()
    print("Load with:")
    print(f"  data  = np.load('{save_path}', allow_pickle=True).item()")
    print("  folds = data['folds']")

    return folds
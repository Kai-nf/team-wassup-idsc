"""
train_svm_versatile.py
=======================
Versatile SVM (RBF kernel) ablation script for Brugada ECG classification.
Supports four dataset versions via --version argument.

Version overview:
  v1   : dataset_v1_raw_drop14.npy
         .npy contains a pickled dict with embedded fold splits (pre-built).
         Split source: embedded in file (may differ from drop14 manifest).

  v2   : dataset_v2_filtered_drop14.npy
         Same dict-with-folds structure as v1.
         Split source: embedded in file (may differ from drop14 manifest).

  v3.2 : dataset_v3.2_wavelet.npy  (N, 101, 12) patient-level median beats
         Split source: fold_composition_v3.2.json  (beat_indices per fold)

  v4   : dataset_v4_features_drop14.csv  tabular feature matrix
         Split source: fold_composition_v4_drop14.json  (indices per fold)

Pipeline per fold:
  1. Load & split         (version-specific logic)
  2. Flatten to 2D        (if input is 3D)
  3. StandardScaler       (fit on X_train only)
  4. PCA(0.95 variance)   (applied when input was 3D — reduces p >> n risk)
  5. SVC(rbf, balanced)   (fit on scaled/reduced train set)
  6. Patient-level metrics (macro_f1, precision, recall, specificity, roc_auc, pr_auc)
  7. JSON output          results/svm_ablation_{version}.json

Clinical note:
  SVM-RBF on flattened ECG beats is a well-established baseline for Brugada
  Type 1 detection. class_weight='balanced' mirrors pos_weight in the PyTorch
  scripts, correcting for the ~21% minority class prevalence.
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# === CONFIG ===
# =============================================================================

RANDOM_SEED    = 42
THRESHOLD      = 0.35
PCA_VARIANCE   = 0.95   # fraction of variance retained when PCA is applied
SMOTE_SAMPLING_STRATEGY = 0.6
SMOTE_K_NEIGHBORS       = 2

THIS_DIR = Path(__file__).resolve().parent

VERSION_CONFIG = {
    "v1": {
        "wavelet_path":  THIS_DIR / "JiaKang" / "dataset_v1_raw_drop14.npy",
        "manifest_path": None,   # splits embedded in .npy dict
        "output_file":   "svm_ablation_v1.json",
        "split_source":  "embedded_in_npy",
        "data_format":   "dict_npy",
    },
    "v2": {
        "wavelet_path":  THIS_DIR / "JiaKang" / "dataset_v2_filtered_drop14.npy",
        "manifest_path": None,
        "output_file":   "svm_ablation_v2.json",
        "split_source":  "embedded_in_npy",
        "data_format":   "dict_npy",
    },
    "v3.2": {
        "wavelet_path":  THIS_DIR.parents[1] / "JiaKang" / "dataset_v3.2_wavelet.npy",
        "manifest_path": THIS_DIR.parents[1] / "JiaKang" / "fold_composition_v3.2.json",
        "output_file":   "method3.2_svm_smote.json",
        "split_source":  "fold_composition_v3.2.json",
        "data_format":   "tensor_npy_with_json",
    },
    "v4": {
        "wavelet_path":  THIS_DIR.parents[1] / "Steve" / "dataset_v4_features_drop14.csv",
        "manifest_path": THIS_DIR.parents[1] / "Steve" / "fold_composition_v4_drop14.json",
        "output_file":   "method4_svm_smote.json",
        "split_source":  "fold_composition_v4_drop14.json",
        "data_format":   "csv_with_json",
    },
}

OUTPUT_DIR = Path("results")


import json
import numpy as np

# Add this custom class to translate NumPy types to standard Python types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)
# =============================================================================
# PATH DEBUG
# =============================================================================

def debug_paths(cfg: dict) -> None:
    """Print resolved paths and abort if any required file is missing."""
    print("=" * 64)
    print("PATH RESOLUTION CHECK  (version={})".format(cfg["version"]))
    print("=" * 64)
    print("Script location : {}".format(THIS_DIR))
    print()

    to_check = {"Data file": cfg["wavelet_path"]}
    if cfg["manifest_path"] is not None:
        to_check["Fold manifest"] = cfg["manifest_path"]

    any_missing = False
    for label, p in to_check.items():
        status = "OK     " if Path(p).exists() else "MISSING"
        print("  [{}]  {}".format(status, p))
        if not Path(p).exists():
            any_missing = True
    print()

    if any_missing:
        raise FileNotFoundError(
            "\n[ABORT] One or more required files are missing (see MISSING above)."
        )
    print("All files found. Proceeding.\n")


# =============================================================================
# DATA LOADING — VERSION-SPECIFIC
# =============================================================================

def load_v1_v2(cfg: dict):
    """
    Load v1 or v2 dataset.

    These .npy files contain a pickled Python dict with pre-built fold splits.
    allow_pickle=True is required — without it np.load raises ValueError.

    Expected structure (either of these common patterns):
        {"folds": [{"X_train": ..., "X_test": ..., "y_train": ..., "y_test": ...}, ...]}
      OR
        {"folds": [{"train": {"X": ..., "y": ...}, "test": {"X": ..., "y": ...}}, ...]}

    Returns
    -------
    folds_data : list of dicts, each with keys X_train, X_test, y_train, y_test
    split_note : str  warning about fold split provenance
    """
    raw = np.load(str(cfg["wavelet_path"]), allow_pickle=True)

    # np.load on a dict-npy returns a 0-d object array — extract the dict
    if isinstance(raw, np.ndarray) and raw.ndim == 0:
        data = raw.item()
    elif isinstance(raw, dict):
        data = raw
    else:
        raise ValueError(
            "[ERROR] Unexpected structure in {}. "
            "Expected a pickled dict, got {}.".format(cfg["wavelet_path"], type(raw))
        )

    if "folds" not in data:
        raise KeyError(
            "[ERROR] 'folds' key not found in {}. "
            "Available keys: {}".format(cfg["wavelet_path"], list(data.keys()))
        )

    raw_folds = data["folds"]
    folds_data = []

    for i, fold in enumerate(raw_folds):
        # Handle both common storage patterns
        if "X_train" in fold:
            entry = {
                "X_train": np.array(fold["X_train"]),
                "X_test":  np.array(fold["X_test"]),
                "y_train": np.array(fold["y_train"]),
                "y_test":  np.array(fold["y_test"]),
            }
        elif "train" in fold and "X" in fold["train"]:
            entry = {
                "X_train": np.array(fold["train"]["X"]),
                "X_test":  np.array(fold["test"]["X"]),
                "y_train": np.array(fold["train"]["y"]),
                "y_test":  np.array(fold["test"]["y"]),
            }
        else:
            raise KeyError(
                "[ERROR] Cannot parse fold {} structure. "
                "Keys found: {}".format(i, list(fold.keys()))
            )
        folds_data.append(entry)

    split_note = (
        "WARNING: v1/v2 fold splits are embedded in the .npy file. "
        "These may not match master_folds_drop14.json patient splits. "
        "Ablation comparison with v3.2/v4 is approximate."
    )
    return folds_data, split_note


def load_v3_2(cfg: dict):
    """
    Load v3.2 patient-level wavelet dataset.

    Wavelet array  : (N, 101, 12)  float32
    Fold manifest  : fold_composition_v3.2.json
                     folds[k][train/test][beat_indices] -> row indices into array

    Returns
    -------
    folds_data : list of dicts with X_train, X_test, y_train, y_test
    split_note : str
    """
    X = np.load(str(cfg["wavelet_path"]))

    with open(str(cfg["manifest_path"]), "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for key in ("labels", "folds"):
        if key not in manifest:
            raise KeyError(
                "[ERROR] '{}' key missing in fold_composition_v3.2.json.".format(key)
            )

    labels = np.array(manifest["labels"], dtype=np.int8)
    folds  = manifest["folds"]

    folds_data = []
    for fold_key in sorted(folds.keys(), key=lambda k: int(k)):
        fold = folds[fold_key]
        train_idx = np.array(fold["train"]["beat_indices"], dtype=int)
        test_idx  = np.array(fold["test"]["beat_indices"],  dtype=int)
        folds_data.append({
            "X_train": X[train_idx],
            "X_test":  X[test_idx],
            "y_train": labels[train_idx],
            "y_test":  labels[test_idx],
            "fold_key": fold_key,
        })

    return folds_data, "fold_composition_v3.2.json (beat_indices)"


def load_v4(cfg: dict):
    """
    Load v4 tabular feature dataset.

    CSV         : rows = patients, columns = features + patient_id + label
    Fold JSON   : fold_composition_v4_drop14.json
                  folds[k][train/test][indices] -> row indices into dataframe

    patient_id and label columns are excluded from X features.

    Returns
    -------
    folds_data : list of dicts with X_train, X_test, y_train, y_test
    split_note : str
    """
    df = pd.read_csv(str(cfg["wavelet_path"]))

    # Identify feature columns — exclude patient_id and label
    exclude_cols = {"patient_id", "label"}
    feature_cols = [c for c in df.columns if c.lower() not in exclude_cols]
    if "label" not in [c.lower() for c in df.columns]:
        raise KeyError(
            "[ERROR] 'label' column not found in {}. "
            "Available columns: {}".format(cfg["wavelet_path"], list(df.columns))
        )

    # Get label column with case-insensitive lookup
    label_col = next(c for c in df.columns if c.lower() == "label")
    X_full = df[feature_cols].to_numpy(dtype=np.float32)
    y_full = df[label_col].to_numpy(dtype=np.int8)

    with open(str(cfg["manifest_path"]), "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if "folds" not in manifest:
        raise KeyError("[ERROR] 'folds' key missing in fold_composition_v4_drop14.json.")

    folds_data = []
    for fold_key in sorted(manifest["folds"].keys(), key=lambda k: int(k)):
        fold = manifest["folds"][fold_key]

        # Defensive key lookup — v4 uses 'indices'; raise clearly if missing
        for split in ("train", "test"):
            if "indices" not in fold[split]:
                raise KeyError(
                    "[ERROR] fold {} {} has no 'indices' key. "
                    "Available keys: {}".format(fold_key, split, list(fold[split].keys()))
                )

        train_idx = np.array(fold["train"]["indices"], dtype=int)
        test_idx  = np.array(fold["test"]["indices"],  dtype=int)
        folds_data.append({
            "X_train": X_full[train_idx],
            "X_test":  X_full[test_idx],
            "y_train": y_full[train_idx],
            "y_test":  y_full[test_idx],
            "fold_key": fold_key,
            "feature_cols": feature_cols,
        })

    return folds_data, "fold_composition_v4_drop14.json (indices)"


def load_dataset(cfg: dict):
    """Dispatcher: routes to the correct loader based on data_format."""
    fmt = cfg["data_format"]
    if fmt == "dict_npy":
        return load_v1_v2(cfg)
    elif fmt == "tensor_npy_with_json":
        return load_v3_2(cfg)
    elif fmt == "csv_with_json":
        return load_v4(cfg)
    else:
        raise ValueError("Unknown data_format: {}".format(fmt))


# =============================================================================
# PREPROCESSING PIPELINE
# =============================================================================

def flatten_if_3d(X: np.ndarray, label: str) -> np.ndarray:
    """
    Flatten (N, T, L) → (N, T*L) if input is 3D, leave 2D unchanged.
    Prints shape before and after for architecture verification.
    """
    if X.ndim == 3:
        N, T, L = X.shape
        X_flat = X.reshape(N, T * L)
        print("  [{}] Flatten: {} → {}  ({}×{} = {} features)".format(
            label, X.shape, X_flat.shape, T, L, T * L))
        return X_flat
    else:
        print("  [{}] Already 2D: {}  (no flatten needed)".format(label, X.shape))
        return X


def preprocess_fold(X_train: np.ndarray, X_test: np.ndarray,
                    was_3d: bool) -> tuple:
    """
    Full preprocessing pipeline for one fold:
        StandardScaler (fit on train only)
        → PCA(0.95 variance, only if input was 3D)

    Returns
    -------
    X_train_proc : np.ndarray  processed training features
    X_test_proc  : np.ndarray  processed test features
    n_components : int or None  PCA components selected (None if PCA skipped)
    """
    # ── StandardScaler ───────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── PCA (only for originally-3D inputs to address p >> n) ────────────────
    if was_3d:
        pca = PCA(n_components=PCA_VARIANCE, random_state=RANDOM_SEED)
        X_train_p = pca.fit_transform(X_train_s)
        X_test_p  = pca.transform(X_test_s)
        n_components = pca.n_components_
        print("  [PCA] {} → {} components  ({:.1f}% variance retained)".format(
            X_train_s.shape[1], n_components,
            pca.explained_variance_ratio_.sum() * 100))
        return X_train_p, X_test_p, n_components
    else:
        return X_train_s, X_test_s, None


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: np.ndarray) -> dict | None:
    """
    Compute all six benchmark metrics. Returns None for single-class folds.
    """
    if len(np.unique(y_true)) < 2:
        return None

    macro_f1  = f1_score(y_true, y_pred, average="macro")
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    roc_auc   = roc_auc_score(y_true, y_prob)
    pr_auc    = average_precision_score(y_true, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity    = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "macro_f1":    round(float(macro_f1),    4),
        "precision":   round(float(precision),   4),
        "recall":      round(float(recall),      4),
        "specificity": round(float(specificity), 4),
        "roc_auc":     round(float(roc_auc),     4),
        "pr_auc":      round(float(pr_auc),      4),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Versatile SVM ablation across four Brugada ECG dataset versions."
    )
    parser.add_argument(
        "--version", choices=["v1", "v2", "v3.2", "v4"], required=True,
        help="Dataset version: v1, v2, v3.2, or v4."
    )
    args = parser.parse_args()

    cfg = {**VERSION_CONFIG[args.version], "version": args.version}

    # ── Path validation ───────────────────────────────────────────────────────
    debug_paths(cfg)

    # ── Load dataset ──────────────────────────────────────────────────────────
    folds_data, split_note = load_dataset(cfg)
    n_folds = len(folds_data)

    # Peek at the first fold to determine dimensionality
    first_X = folds_data[0]["X_train"]
    was_3d  = first_X.ndim == 3
    input_shape = first_X.shape[1:]  # (101, 12) or (n_features,)

    print("Version              : {}".format(args.version))
    print("Folds loaded         : {}".format(n_folds))
    print("Input shape per sample: {}".format(input_shape))
    print("3D input (will PCA)  : {}".format(was_3d))
    print("Split source         : {}".format(split_note))
    print()
    if "WARNING" in split_note:
        print("[!] {}".format(split_note))
        print()

    # ── Output directory ──────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


    smote_applied = (args.version == "v4")
 
    print("Version              : {}".format(args.version))
    print("Folds loaded         : {}".format(n_folds))
    print("Input shape per sample: {}".format(input_shape))
    print("3D input (will PCA)  : {}".format(was_3d))
    print("SMOTE active         : {}  (v4 only -- tabular features only)".format(smote_applied))
    print("Split source         : {}".format(split_note))
    print()
    if "WARNING" in split_note:
        print("[!] {}".format(split_note))
        print()

    # ── Results container ─────────────────────────────────────────────────────
    results = {
        "method":        "svm_ablation",
        "model":         "svc_rbf",
        "version":       args.version,
        "split_source":  split_note,
        "folds":         {},
        "summary":       {},
        "config": {
            "kernel":         "rbf",
            "class_weight":   "balanced",
            "probability":    True,
            "random_state":   RANDOM_SEED,
            "threshold":      THRESHOLD,
            "pca_variance":   PCA_VARIANCE if was_3d else "skipped_2d_input",
            "input_shape":    list(input_shape),
            "was_3d_input":   was_3d,
            "smote_applied":  smote_applied,
        },
    }

    metric_keys = ["macro_f1", "precision", "recall", "specificity", "roc_auc", "pr_auc"]
    collected   = {k: [] for k in metric_keys}

    # ── Cross-validation loop ─────────────────────────────────────────────────
    for fold_idx, fold in enumerate(folds_data):
        fold_key = fold.get("fold_key", str(fold_idx))

        X_train_raw = fold["X_train"].astype(np.float32)
        X_test_raw  = fold["X_test"].astype(np.float32)
        y_train     = fold["y_train"]
        y_test      = fold["y_test"]

        n_train = len(X_train_raw)
        n_test  = len(X_test_raw)
        n_test_pos = int(y_test.sum())

        print("[Fold {}]  train={}, test={}, test_pos={}".format(
            fold_key, n_train, n_test, n_test_pos))

        # ── 1. Flatten ────────────────────────────────────────────────────────
        X_train_f = flatten_if_3d(X_train_raw, "train")
        X_test_f  = flatten_if_3d(X_test_raw,  "test")

        # ── 2. Scale + optional PCA ───────────────────────────────────────────
        X_train_p, X_test_p, n_pca = preprocess_fold(X_train_f, X_test_f, was_3d)
        
        # Step 3 -- SMOTE (v4 only, training set only, never test)
        if smote_applied:
            smote = SMOTE(random_state=RANDOM_SEED)
            X_train_p, y_train = smote.fit_resample(X_train_p, y_train)
            unique, counts = np.unique(y_train, return_counts=True)
            dist_str = ", ".join("{}={}".format(int(u), int(c))
                                 for u, c in zip(unique, counts))
            print("  [SMOTE] Training labels resampled to: {}".format(dist_str))

        # ── 4. Train SVM ──────────────────────────────────────────────────────
        svm = SVC(
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=RANDOM_SEED,
        )
        svm.fit(X_train_p, y_train)

        # ── 5. Predict ────────────────────────────────────────────────────────
        y_pred = svm.predict(X_test_p)
        y_prob = svm.predict_proba(X_test_p)[:, 1]  # probability of Brugada class

        # ── 6. Metrics ────────────────────────────────────────────────────────
        if len(np.unique(y_test)) < 2:
            print("  [WARN] Fold {} has only one class — skipping metrics.".format(fold_key))
            results["folds"][fold_key] = {
                "macro_f1": None, "precision": None, "recall": None,
                "specificity": None, "roc_auc": None, "pr_auc": None,
                "TP": None, "TN": None, "FP": None, "FN": None,
                "n_train": n_train, "n_test": n_test,
                "n_test_positive": n_test_pos,
                "pca_components": n_pca,
            }
            continue

        m = compute_metrics(y_test, y_pred, y_prob)

        print("  macro_f1={macro_f1:.4f}  precision={precision:.4f}  "
              "recall={recall:.4f}  specificity={specificity:.4f}  "
              "roc_auc={roc_auc:.4f}  pr_auc={pr_auc:.4f}".format(**m))

        results["folds"][fold_key] = {
            **m,
            "n_train":        n_train,
            "n_test":         n_test,
            "n_test_positive": n_test_pos,
            "pca_components": n_pca,
        }

        for k in metric_keys:
            collected[k].append(m[k])

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = {}
    print("\n=== CROSS-VALIDATION SUMMARY  [version={}] ===".format(args.version))
    for k in metric_keys:
        vals = collected[k]
        mean = float(np.mean(vals)) if vals else 0.0
        std  = float(np.std(vals))  if vals else 0.0
        summary["{}_mean".format(k)] = round(mean, 4)
        summary["{}_std".format(k)]  = round(std,  4)
        print("  {:<14}: {:.4f} +/- {:.4f}".format(k, mean, std))

    results["summary"] = summary

    # ── Write JSON ────────────────────────────────────────────────────────────
    out_path = OUTPUT_DIR / cfg["output_file"]
    with open(str(out_path), "w", encoding="utf-8") as f:
        # Add the cls argument here!
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    print("\nResults saved to {}".format(out_path))


if __name__ == "__main__":
    main()
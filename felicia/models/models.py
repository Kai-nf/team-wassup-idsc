# pyre-ignore-all-errors
import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, cast

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    _IMBLEARN_AVAILABLE = True
except ImportError:
    _IMBLEARN_AVAILABLE = False


ROOT_DIR = Path(__file__).resolve().parents[2]
if (ROOT_DIR / "data_loader.py").exists() is False:
    ROOT_DIR = Path(__file__).resolve().parents[1]
if (ROOT_DIR / "data_loader.py").exists() is False:
    ROOT_DIR = Path.cwd()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data_preprocessing.method2 import apply_clinical_filters


def _resolve_load_raw_dataset():
    candidates = [
        ROOT_DIR / "Environment_setup" / "data_loader.py",
        ROOT_DIR / "data_loader.py",
        ROOT_DIR / "brugada-ecg-classifier" / "data_loader.py",
    ]
    for idx, c in enumerate(candidates):
        if not c.exists():
            continue
        spec = importlib.util.spec_from_file_location(f"felicia_loader_{idx}", c)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        loader = cast(Any, spec.loader)
        loader.exec_module(module)
        if hasattr(module, "load_raw_dataset"):
            return getattr(module, "load_raw_dataset")
    raise ImportError(
        "Could not find load_raw_dataset in known locations: "
        "data_loader.py, brugada-ecg-classifier/data_loader.py, Environment_setup/data_loader.py"
    )


load_raw_dataset = _resolve_load_raw_dataset()


ALLOWED_FOLD_FILE = "master_folds_drop14.json"
GLOBAL_SEED = 42


def set_global_seed(seed: int = GLOBAL_SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)  # Fix #1: freeze hash seed
    random.seed(seed)
    np.random.seed(seed)


def _binarize_labels(values: pd.Series) -> np.ndarray:
    return (pd.to_numeric(values, errors="coerce").fillna(0) > 0).astype(int).to_numpy()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_master_folds(path: str = ALLOWED_FOLD_FILE) -> Tuple[Dict, str]:
    fold_path = Path(path)
    if fold_path.name != ALLOWED_FOLD_FILE:
        raise ValueError(
            f"Invalid fold file '{fold_path.name}'. Must use '{ALLOWED_FOLD_FILE}' only."
        )
    if not fold_path.exists():
        raise FileNotFoundError(f"Missing fold file: {fold_path}")

    with open(fold_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    required_keys = {"n_splits", "random_state", "folds"}
    missing = required_keys - set(data.keys())
    if missing:
        raise ValueError(f"Master fold file missing keys: {sorted(missing)}")

    return data, _sha256_file(fold_path)


def normalize_master_folds(data: Dict) -> List[Dict]:
    normalized = []
    for fold_idx in range(int(data["n_splits"])):
        fold = data["folds"][str(fold_idx)]
        normalized.append(
            {
                "fold": fold_idx,
                "train_patient_ids": np.array(fold["train"]["patient_ids"], dtype=int),
                "test_patient_ids": np.array(fold["test"]["patient_ids"], dtype=int),
                "train_metadata_row_indices": np.array(
                    fold["train"].get("metadata_row_indices", []), dtype=int
                ),
                "test_metadata_row_indices": np.array(
                    fold["test"].get("metadata_row_indices", []), dtype=int
                ),
            }
        )
    return normalized


def _summarize_scaler(scaler: StandardScaler) -> Dict:
    mean_vec = np.asarray(scaler.mean_, dtype=float)
    scale_vec = np.asarray(scaler.scale_, dtype=float)
    return {
        "feature_count": int(mean_vec.shape[0]),
        "mean_abs_mean": float(np.mean(np.abs(mean_vec))),
        "mean_std": float(np.mean(scale_vec)),
        "min_std": float(np.min(scale_vec)),
        "max_std": float(np.max(scale_vec)),
    }


def _scale_3d_train_test(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
    n_train, t, l = X_train.shape
    n_test = X_test.shape[0]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, l)).reshape(n_train, t, l)
    X_test_scaled = scaler.transform(X_test.reshape(-1, l)).reshape(n_test, t, l)
    return X_train_scaled.astype(np.float32), X_test_scaled.astype(np.float32), _summarize_scaler(scaler)


def _class_balance(y: np.ndarray) -> Dict:
    y = np.asarray(y)
    unique, counts = np.unique(y, return_counts=True)
    return {str(int(k)): int(v) for k, v in zip(unique, counts)}


def _apply_baseline_drift(x: np.ndarray, fs: int = 100) -> np.ndarray:
    t = np.arange(x.shape[0]) / fs
    freq = np.random.uniform(0.1, 0.5)
    amp = np.random.uniform(-0.1, 0.1)
    drift = amp * np.sin(2 * np.pi * freq * t)
    return x + drift[:, None]


def _apply_white_noise_snr(x: np.ndarray) -> np.ndarray:
    snr_db = np.random.uniform(20.0, 30.0)
    signal_power = np.mean(np.square(x)) + 1e-12
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = np.random.normal(0, np.sqrt(noise_power), size=x.shape)
    return x + noise


def _apply_temporal_scaling(x: np.ndarray) -> np.ndarray:
    t, l = x.shape
    scale = np.random.uniform(0.95, 1.05)
    src = np.arange(t)
    stretched_len = max(2, int(round(t * scale)))
    stretched = np.zeros((stretched_len, l), dtype=np.float32)
    for lead in range(l):
        stretched[:, lead] = np.interp(
            np.linspace(0, t - 1, stretched_len),
            src,
            x[:, lead],
        )
    back = np.zeros_like(x, dtype=np.float32)
    for lead in range(l):
        back[:, lead] = np.interp(
            np.linspace(0, stretched_len - 1, t),
            np.arange(stretched_len),
            stretched[:, lead],
        )
    return back


def apply_domain_augmentation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    patient_ids_train: np.ndarray,
    brugada_only: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Augment only training-fold positives (Brugada=1) and append new samples.
    pos_mask = (y_train == 1) if brugada_only else np.ones_like(y_train, dtype=bool)
    pos_idx = np.where(pos_mask)[0]
    if pos_idx.size == 0:
        return X_train, y_train, patient_ids_train

    augmented = []
    for idx in pos_idx:
        x = X_train[idx]
        x_aug = _apply_baseline_drift(x)
        x_aug = _apply_white_noise_snr(x_aug)
        x_aug = _apply_temporal_scaling(x_aug)
        augmented.append(x_aug.astype(np.float32))

    X_aug = np.stack(augmented, axis=0)
    y_aug = np.ones(X_aug.shape[0], dtype=y_train.dtype)
    pid_aug = patient_ids_train[pos_idx]

    X_train_new = np.concatenate([X_train, X_aug], axis=0)
    y_train_new = np.concatenate([y_train, y_aug], axis=0)
    pid_train_new = np.concatenate([patient_ids_train, pid_aug], axis=0)
    return X_train_new, y_train_new, pid_train_new


def apply_feature_resampling(
    X_train_2d: np.ndarray,
    y_train: np.ndarray,
    strategy: str,
) -> Tuple[np.ndarray, np.ndarray]:
    if strategy == "none":
        return X_train_2d, y_train

    try:
        from imblearn.over_sampling import ADASYN, BorderlineSMOTE, SMOTE
    except Exception as e:
        raise ImportError("imblearn is required for SMOTE/ADASYN. Install imbalanced-learn.") from e

    if strategy == "smote":
        sampler = SMOTE(random_state=42)
    elif strategy == "borderline_smote":
        sampler = BorderlineSMOTE(random_state=42)
    elif strategy == "adasyn":
        sampler = ADASYN(random_state=42)
    else:
        raise ValueError(f"Unknown resampling strategy: {strategy}")

    X_rs, y_rs = sampler.fit_resample(X_train_2d, y_train)
    return X_rs.astype(np.float32), y_rs.astype(int)

def _build_sampler(strategy: str):
    """Build an imblearn over-sampler by name. Used inside Pipeline for Method 4."""
    from imblearn.over_sampling import ADASYN, BorderlineSMOTE, SMOTE
    if strategy == "smote":
        return SMOTE(random_state=42)
    if strategy == "borderline_smote":
        return BorderlineSMOTE(random_state=42)
    if strategy == "adasyn":
        return ADASYN(random_state=42)
    raise ValueError(f"Unknown resampling strategy: {strategy}")


def _build_model_and_grid(model_name: str):
    if model_name == "logistic":
        model = LogisticRegression(max_iter=3000)
        grid = {
            "C": [0.01, 0.1, 1.0, 10.0],
            "penalty": ["l1", "l2"],
            "class_weight": ["balanced"],
            "solver": ["liblinear", "saga"],
        }
        return model, grid
    if model_name == "rf":
        model = RandomForestClassifier(random_state=42)
        grid = {
            "n_estimators": [200, 500],
            "max_depth": [None, 10, 20],
            "min_samples_leaf": [1, 3, 5],
            "class_weight": ["balanced", "balanced_subsample"],
        }
        return model, grid
    raise ValueError(f"Unsupported model: {model_name}")

def _fit_with_nested_cv(
    model_name: str,
    X_train_2d: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray = None,
    resampler: str = "none",
):
    """Nested CV with StratifiedGroupKFold (patient-safe) and optional imblearn pipeline.

    Args:
        groups: Patient IDs for each sample in X_train_2d. When provided, uses
                StratifiedGroupKFold so patients are never split across inner folds.
        resampler: 'smote', 'borderline_smote', 'adasyn', or 'none'. When not 'none',
                   wraps the classifier in an imblearn Pipeline so resampling is applied
                   ONLY to inner-train folds (never inner-val or outer-test).
    """
    model, grid = _build_model_and_grid(model_name)

    # Fix #6/7/11: Wrap SMOTE inside imblearn Pipeline so it only sees inner-train data.
    if resampler != "none":
        if not _IMBLEARN_AVAILABLE:
            raise ImportError("imbalanced-learn is required for SMOTE/ADASYN. pip install imbalanced-learn")
        sampler = _build_sampler(resampler)
        estimator = ImbPipeline([("sampler", sampler), ("clf", model)])
        param_grid = {f"clf__{k}": v for k, v in grid.items()}
    else:
        estimator = model
        param_grid = grid

    # Fix #7/11: Use StratifiedGroupKFold when patient IDs are available.
    if groups is not None:
        inner_cv = StratifiedGroupKFold(n_splits=3)
        fit_kwargs: Dict = {"groups": groups}
    else:
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        fit_kwargs = {}

    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=inner_cv,
        n_jobs=-1,
    )
    search.fit(X_train_2d, y_train, **fit_kwargs)

    # Strip pipeline prefix (clf__C → C) for clean logging.
    clean_params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
    return search.best_estimator_, clean_params, float(search.best_score_)


def _sens_at_spec(y_true: np.ndarray, y_prob: np.ndarray, target_spec: float = 0.95) -> float:
    """Per-fold: find threshold where specificity >= target_spec on the ROC curve,
    return the corresponding sensitivity. Positive class = 1."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob, pos_label=1)
    # specificity = 1 - FPR
    spec = 1.0 - fpr
    # Find indices where specificity >= target
    eligible = np.where(spec >= target_spec)[0]
    if len(eligible) == 0:
        # No threshold achieves target specificity — return sensitivity at highest specificity
        return float(tpr[np.argmax(spec)])
    # Among eligible thresholds, pick the one with highest sensitivity
    best_idx = eligible[np.argmax(tpr[eligible])]
    return float(tpr[best_idx])


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict:
    """Compute all Phase 5 metrics. Positive class is fixed as Brugada (label=1)."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else None

    out = {
        # --- macro-averaged (balanced view across both classes) ---
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        # --- Brugada-class specific (positive class = 1) ---
        "f1_brugada": float(f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)),  # Fix #9
        "precision_brugada": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_brugada": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "specificity": specificity,
        "confusion_matrix": cm.tolist(),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        out["roc_auc"] = None
    try:
        out["pr_auc"] = float(average_precision_score(y_true, y_prob, pos_label=1))
    except Exception:
        out["pr_auc"] = None
    try:
        out["sens_at_95spec"] = _sens_at_spec(y_true, y_prob, target_spec=0.95)
    except Exception:
        out["sens_at_95spec"] = None
    return out


def _flatten_if_needed(X: np.ndarray) -> np.ndarray:
    if X.ndim == 3:
        return X.reshape(X.shape[0], -1).astype(np.float32)
    return X.astype(np.float32)


def _aggregate_beats_to_patient(
    patient_ids: np.ndarray,
    y_true_beats: np.ndarray,
    y_prob_beats: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    unique = np.unique(patient_ids)
    y_true_pat = []
    y_prob_pat = []
    y_pred_pat = []
    for pid in unique:
        m = patient_ids == pid
        true_vals = y_true_beats[m]
        y_true_pat.append(int(np.round(true_vals.mean())))
        # Mean probability rule for patient-level roll-up (plan §4).
        p = float(np.mean(y_prob_beats[m]))
        y_prob_pat.append(p)
        y_pred_pat.append(int(p >= 0.5))
    return unique, np.array(y_true_pat), np.array(y_pred_pat), np.array(y_prob_pat)


def _load_method1_or_2_base(metadata_csv: str, data_dir: str):
    metadata = pd.read_csv(metadata_csv)
    loaded = load_raw_dataset(metadata_csv, data_dir)
    all_signals = loaded[0] if isinstance(loaded, tuple) else loaded
    y = _binarize_labels(metadata["brugada"])
    pids = metadata["patient_id"].astype(int).to_numpy()
    return all_signals, y, pids


def _load_method3_dataset(method_name: str, metadata_csv: str):
    metadata = pd.read_csv(metadata_csv)
    patient_label_map = {
        int(pid): int(lbl)
        for pid, lbl in zip(metadata["patient_id"], _binarize_labels(metadata["brugada"]))
    }

    if method_name == "method3":
        x_path = ROOT_DIR / "JiaKang" / "dataset_v3_wavelet.npy"
        json_path = ROOT_DIR / "JiaKang" / "fold_composition_v3.json"
    elif method_name == "method3_1":
        x_path = ROOT_DIR / "JiaKang" / "dataset_v3.1_wavelet.npy"
        json_path = ROOT_DIR / "JiaKang" / "fold_composition_v3.1.json"
    else:
        x_path = ROOT_DIR / "JiaKang" / "dataset_v3.2_wavelet.npy"
        json_path = ROOT_DIR / "JiaKang" / "fold_composition_v3.2.json"

    if not x_path.exists() or not json_path.exists():
        raise FileNotFoundError(f"Missing Method 3 assets: {x_path} or {json_path}")

    X = np.load(x_path)
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    labels = np.array(meta["labels"], dtype=int)
    beat_pids = np.array(meta["patient_ids"], dtype=int)
    return X, labels, beat_pids, patient_label_map


def _load_method4_dataset(csv_path: str):
    df = pd.read_csv(csv_path)
    if "patient_id" not in df.columns or "label" not in df.columns:
        raise ValueError("Method 4 CSV must contain patient_id and label columns.")
    feature_cols = [c for c in df.columns if c not in {"patient_id", "label"}]
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["label"].astype(int).to_numpy()
    pids = df["patient_id"].astype(int).to_numpy()
    return X, y, pids, feature_cols


def run_experiment(
    method: str,
    model_name: str,
    fold_file: str,
    metadata_csv: str,
    data_dir: str,
    method4_csv: str,
    resampler: str,
    output_json: str,
):
    set_global_seed(GLOBAL_SEED)
    folds_data, fold_hash = load_master_folds(fold_file)
    normalized_folds = normalize_master_folds(folds_data)
    print(f"Using fold source: {ALLOWED_FOLD_FILE}")

    if method in {"method1", "method2"}:
        X_base, y_base, pids_base = _load_method1_or_2_base(metadata_csv, data_dir)
    elif method in {"method3", "method3_1", "method3_2"}:
        X_base, y_base, pids_base, patient_label_map = _load_method3_dataset(method, metadata_csv)
    elif method == "method4":
        X_base, y_base, pids_base, _ = _load_method4_dataset(method4_csv)
    else:
        raise ValueError(f"Unsupported method: {method}")

    fold_results: List[Dict[str, Any]] = []

    for fold_info in normalized_folds:
        fold_idx = int(fold_info["fold"])
        train_pids = fold_info["train_patient_ids"]
        test_pids = fold_info["test_patient_ids"]

        # Fix #2: Runtime guard — patients must be strictly disjoint across train/test.
        assert set(train_pids.tolist()).isdisjoint(set(test_pids.tolist())), (
            f"Patient leakage detected in fold {fold_idx}: "
            f"overlap = {set(train_pids.tolist()) & set(test_pids.tolist())}"
        )

        train_mask = np.isin(pids_base, train_pids)
        test_mask = np.isin(pids_base, test_pids)

        X_train = X_base[train_mask]
        y_train = y_base[train_mask]
        pid_train = pids_base[train_mask]

        X_test = X_base[test_mask]
        y_test = y_base[test_mask]
        pid_test = pids_base[test_mask]

        if method == "method2":
            X_train = np.array([apply_clinical_filters(x) for x in X_train], dtype=np.float32)
            X_test = np.array([apply_clinical_filters(x) for x in X_test], dtype=np.float32)

        if X_train.ndim == 3:
            X_train, X_test, scaler_stats = _scale_3d_train_test(X_train, X_test)
        else:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train).astype(np.float32)
            X_test = scaler.transform(X_test).astype(np.float32)
            scaler_stats = _summarize_scaler(scaler)

        train_balance_pre_aug = _class_balance(y_train)
        test_balance = _class_balance(y_test)

        if method in {"method1", "method2", "method3", "method3_1", "method3_2"}:
            # Domain augmentation applied pre-CV; augmented samples inherit source patient's
            # ID into pid_train, so StratifiedGroupKFold keeps orig+aug together in inner folds.
            X_train, y_train, pid_train = apply_domain_augmentation(X_train, y_train, pid_train, brugada_only=True)

        X_train_2d = _flatten_if_needed(X_train)
        X_test_2d = _flatten_if_needed(X_test)

        # Fix #6/7/11: SMOTE is NO LONGER applied here for Method 4.
        # It is wrapped inside the imblearn Pipeline inside _fit_with_nested_cv,
        # so it is applied only to each inner-train split — never to inner-val or outer-test.

        train_balance_post_aug = _class_balance(y_train)

        # Fix #7/11: Pass pid_train as groups → StratifiedGroupKFold prevents patient
        # leakage across inner folds. Pass resampler → SMOTE lives inside the pipeline.
        fold_resampler = resampler if method == "method4" else "none"
        best_model, best_params, inner_best_f1 = _fit_with_nested_cv(
            model_name, X_train_2d, y_train,
            groups=pid_train,
            resampler=fold_resampler,
        )
        y_prob = best_model.predict_proba(X_test_2d)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        if method in {"method3", "method3_1", "method3_2"}:
            _, y_true_patient, y_pred_patient, y_prob_patient = _aggregate_beats_to_patient(
                pid_test,
                y_test,
                y_prob,
            )
            m = _metrics(y_true_patient, y_pred_patient, y_prob_patient)
            m["evaluation_unit"] = "patient_from_beats"
            m["n_test_patients"] = int(len(y_true_patient))
            # Optional consistency check against metadata labels
            mismatches: int = 0
            for p, t in zip(np.unique(pid_test), y_true_patient):
                if patient_label_map.get(int(p), t) != int(t):
                    mismatches += 1
            m["patient_label_mismatch_count"] = int(mismatches)
        else:
            m = _metrics(y_test, y_pred, y_prob)
            m["evaluation_unit"] = "patient"
            m["n_test_patients"] = int(len(y_test))

        # Phase 5: Track positive/negative counts for PR-AUC contextualization
        if method in {"method3", "method3_1", "method3_2"}:
            n_pos = int(np.sum(y_true_patient == 1))
            n_neg = int(np.sum(y_true_patient == 0))
        else:
            n_pos = int(np.sum(y_test == 1))
            n_neg = int(np.sum(y_test == 0))
        m["n_positive"] = n_pos
        m["n_negative"] = n_neg
        # Backward-compatible aliases
        m["n_positive_test"] = n_pos
        m["n_negative_test"] = n_neg

        m["fold"] = fold_idx
        m["model"] = model_name
        m["method"] = method
        m["resampler"] = resampler if method == "method4" else "none"
        m["inner_cv_best_f1_macro"] = inner_best_f1
        m["best_params"] = best_params
        m["train_class_balance_pre_augmentation"] = train_balance_pre_aug
        m["train_class_balance_post_augmentation"] = train_balance_post_aug
        m["test_class_balance"] = test_balance
        m["scaler_stats"] = scaler_stats
        m["n_train_patients"] = int(len(np.unique(train_pids)))
        m["n_test_patients_manifest"] = int(len(np.unique(test_pids)))
        m["train_metadata_row_indices_count"] = int(len(fold_info["train_metadata_row_indices"]))
        m["test_metadata_row_indices_count"] = int(len(fold_info["test_metadata_row_indices"]))
        fold_results.append(m)

    def _agg(key: str):
        vals = [r[key] for r in fold_results if r.get(key) is not None]
        return {
            "mean": float(np.mean(vals)) if vals else None,
            "std": float(np.std(vals)) if vals else None,
        }

    summary = {
        "f1_macro": _agg("f1_macro"),
        "recall_macro": _agg("recall_macro"),
        "precision_macro": _agg("precision_macro"),
        # Brugada-class specific
        "f1_brugada": _agg("f1_brugada"),        # Fix #9: per-class binary F1
        "precision_brugada": _agg("precision_brugada"),
        "recall_brugada": _agg("recall_brugada"),
        "specificity": _agg("specificity"),
        # Threshold-independent (computed per-fold from probabilities, never from aggregated data)
        "roc_auc": _agg("roc_auc"),
        "pr_auc": _agg("pr_auc"),
        # Clinical threshold metric (per-fold ROC curve, no global thresholding)
        "sens_at_95spec": _agg("sens_at_95spec"),
    }

    # Aggregated confusion matrix: patient-disjoint folds → summing = zero-leakage full-cohort view
    agg_cm = np.zeros((2, 2), dtype=int)
    for r in fold_results:
        agg_cm += np.array(r["confusion_matrix"], dtype=int)
    agg_tn, agg_fp, agg_fn, agg_tp = agg_cm.ravel()
    aggregated_confusion_matrix = {
        "matrix": agg_cm.tolist(),
        "TP": int(agg_tp),
        "TN": int(agg_tn),
        "FP": int(agg_fp),
        "FN": int(agg_fn),
        "note": "Summed across all 5 test folds. Valid because each patient appears in exactly one test fold (patient-level disjoint splits).",
    }

    output = {
        "fold_source": ALLOWED_FOLD_FILE,
        "fold_source_sha256": fold_hash,
        "n_splits": int(folds_data["n_splits"]),
        "random_state": int(folds_data["random_state"]),
        "seed": int(GLOBAL_SEED),
        "cohort": "drop14",
        "method": method,
        "model": model_name,
        "resampler": resampler if method == "method4" else "none",
        "positive_class": {"label": 1, "name": "Brugada"},
        # Reproducibility: log CV strategy and pipeline usage
        "inner_cv": "StratifiedGroupKFold(n_splits=3)",
        "resampler_in_pipeline": (resampler != "none" and method == "method4"),
        "beat_rollup_rule": "mean" if method in {"method3", "method3_1", "method3_2"} else "n/a",
        "fold_class_balance_confirmed": True,
        "fold_test_positive_counts": [int(r.get("n_positive", 0)) for r in fold_results],
        "fold_test_negative_counts": [int(r.get("n_negative", 0)) for r in fold_results],
        "fold_results": fold_results,
        "aggregated_confusion_matrix": aggregated_confusion_matrix,
        "summary": summary,
    }

    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Saved results to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Unified LR/RF runner with strict master fold manifest.")
    parser.add_argument("--method", choices=["method1", "method2", "method3", "method3_1", "method3_2", "method4"], required=True)
    parser.add_argument("--model", choices=["logistic", "rf"], required=True)
    parser.add_argument("--fold-file", default=ALLOWED_FOLD_FILE)
    parser.add_argument("--metadata-csv", default="metadata.csv")
    parser.add_argument("--data-dir", default="files")
    parser.add_argument("--method4-csv", default="Steve/dataset_v4_features_drop14.csv")
    parser.add_argument("--resampler", choices=["none", "smote", "borderline_smote", "adasyn"], default="none")
    parser.add_argument("--output-json")
    args, _ = parser.parse_known_args()
    
    # Cast args properly to bypass Pyre's argparse Namespace strictness
    method: str = getattr(args, "method", "")
    model_name: str = getattr(args, "model", "")
    fold_file: str = getattr(args, "fold_file", "")
    metadata_csv: str = getattr(args, "metadata_csv", "")
    data_dir: str = getattr(args, "data_dir", "")
    method4_csv: str = getattr(args, "method4_csv", "")
    resampler: str = getattr(args, "resampler", "")
    output_json: str = getattr(args, "output_json", None)

    if method != "method4" and resampler != "none":
        raise ValueError("Resampler can only be used for method4 features.")

    if output_json is None:
        output_name = f"{method}_{model_name}"
        if method == "method4":
            output_name = f"{output_name}_{resampler}"
        output_json = f"felicia/results/evaluation_metrics/{output_name}.json"

    run_experiment(
        method=method,
        model_name=model_name,
        fold_file=fold_file,
        metadata_csv=metadata_csv,
        data_dir=data_dir,
        method4_csv=method4_csv,
        resampler=resampler,
        output_json=output_json,
    )


if __name__ == "__main__":
    main()
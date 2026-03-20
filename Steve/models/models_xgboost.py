"""
model_xgboost.py
================
XGBoost classifier for Brugada Syndrome detection on tabular feature dataset v4.

Usage:
    python model_xgboost.py --resampler adasyn
    python model_xgboost.py --resampler smote
    python model_xgboost.py --resampler borderline_smote
    python model_xgboost.py --resampler none

Resampler options:
    adasyn            ADASYN — adaptive density-based oversampling, focuses
                      synthetic samples on harder boundary cases.
    smote             SMOTE(sampling_strategy=0.6, k_neighbors=2) — tuned
                      parameters from earlier project ablation. Less aggressive
                      than default SMOTE; k=2 keeps synthetic points close to
                      real Brugada patients to avoid majority-class overlap.
    borderline_smote  BorderlineSMOTE — only oversamples minority points near
                      the decision boundary. More conservative than full SMOTE.
    none              No oversampling. scale_pos_weight=neg/pos is set on the
                      XGBClassifier to handle class imbalance directly via
                      loss weighting instead of data augmentation.

Output JSON schema (compatible with versatile_ensemble.py SCHEMA_CONFIG):
    {
      "model":        "xgboost",
      "resampler":    "<chosen>",
      "fold_results": [
        {
          "fold":        int,
          "patient_ids": list[int],
          "y_true":      list[int],
          "y_prob":      list[float],
          "y_pred":      list[int],
          "train_class_balance_pre_resampling":  {"0": int, "1": int},
          "train_class_balance_post_resampling": {"0": int, "1": int},
          "f1_macro":    float,
          "recall":      float,
          "precision":   float,
          "specificity": float,
          "roc_auc":     float
        }, ...
      ]
    }

Output file: results/method4_xgboost_{resampler}.json
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# =============================================================================
# CONFIGURATION
# =============================================================================

RANDOM_SEED = 42

THIS_DIR  = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]      # adjust parents[N] to match repo depth

DATA_PATH  = REPO_ROOT / "Steve"     / "dataset_v4_features_drop14.csv"
FOLDS_PATH = REPO_ROOT               / "master_folds_drop14.json"
OUTPUT_DIR = REPO_ROOT / "results"

# XGBoost hyperparameters
# Shallow trees (max_depth=3) prevent overfitting on n=349.
# subsample and colsample_bytree add stochastic regularisation.
XGBOOST_BASE_PARAMS = dict(
    max_depth        = 3,
    learning_rate    = 0.05,
    n_estimators     = 300,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    random_state     = RANDOM_SEED,
    eval_metric      = "logloss",
    verbosity        = 0,
)

# SMOTE tuned parameters (from project ablation — prevents majority-class overlap)
SMOTE_SAMPLING_STRATEGY = 0.6
SMOTE_K_NEIGHBORS       = 2


# =============================================================================
# RESAMPLER FACTORY
# =============================================================================

RESAMPLER_CHOICES = ["adasyn", "smote", "borderline_smote", "none"]

def build_resampler(name: str):
    """
    Return an instantiated resampler object or None.

    For 'none': returns None — XGBClassifier will use scale_pos_weight instead.
    """
    if name == "adasyn":
        return ADASYN(random_state=RANDOM_SEED)

    elif name == "smote":
        # Tuned parameters from project ablation:
        #   sampling_strategy=0.6 — minority grows to 60% of majority count.
        #     Less aggressive than default 1.0; avoids flooding the feature
        #     space with synthetic points that overlap the majority boundary.
        #   k_neighbors=2 — tight neighbourhood keeps synthetic Brugada samples
        #     close to real Brugada patients, reducing majority-class overlap.
        return SMOTE(
            sampling_strategy=SMOTE_SAMPLING_STRATEGY,
            k_neighbors=SMOTE_K_NEIGHBORS,
            random_state=RANDOM_SEED,
        )

    elif name == "borderline_smote":
        return BorderlineSMOTE(random_state=RANDOM_SEED)

    elif name == "none":
        return None

    else:
        sys.exit(f"[ABORT] Unknown resampler: {name}. Choose from {RESAMPLER_CHOICES}")


# =============================================================================
# PATH VALIDATION
# =============================================================================

def validate_paths() -> None:
    any_missing = False
    print("=" * 62)
    print("PATH RESOLUTION CHECK")
    print("=" * 62)
    for label, p in [("Data (v4 CSV)", DATA_PATH), ("Fold manifest", FOLDS_PATH)]:
        status = "OK     " if p.exists() else "MISSING"
        print(f"  [{status}]  {p}")
        if not p.exists():
            any_missing = True
    print()
    if any_missing:
        sys.exit("[ABORT] One or more required files are missing.")
    print("All files found. Proceeding.\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    # ── Parse arguments ────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="XGBoost Brugada classifier with selectable oversampling."
    )
    parser.add_argument(
        "--resampler",
        choices=RESAMPLER_CHOICES,
        required=True,
        help="Oversampling strategy: adasyn | smote | borderline_smote | none",
    )
    args = parser.parse_args()

    resampler     = build_resampler(args.resampler)
    resampler_str = args.resampler

    # ── Startup summary ────────────────────────────────────────────────────────
    print("=" * 62)
    print("XGBoost  —  Brugada Syndrome Detection  (Dataset v4)")
    print("=" * 62)
    print(f"  Resampler : {resampler_str}")
    if resampler_str == "smote":
        print(f"    sampling_strategy = {SMOTE_SAMPLING_STRATEGY}")
        print(f"    k_neighbors       = {SMOTE_K_NEIGHBORS}")
    if resampler_str == "none":
        print("    scale_pos_weight will be set per fold (neg/pos ratio)")
    print()

    validate_paths()

    # ── Load data ──────────────────────────────────────────────────────────────
    df = pd.read_csv(str(DATA_PATH))
    with open(str(FOLDS_PATH), "r", encoding="utf-8") as f:
        master_folds = json.load(f)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f"method4_xgboost_{resampler_str}.json"

    # ── Results container ──────────────────────────────────────────────────────
    output_data = {
        "model":        "xgboost",
        "resampler":    resampler_str,
        "fold_results": [],
    }

    # ── Cross-validation loop ──────────────────────────────────────────────────
    for fold_id in range(5):
        print(f"[Fold {fold_id}]" + "-" * 50)

        # ── 1. Patient splits ──────────────────────────────────────────────────
        fold_key = str(fold_id)
        train_ids = [int(p) for p in master_folds["folds"][fold_key]["train"]["patient_ids"]]
        test_ids  = [int(p) for p in master_folds["folds"][fold_key]["test"]["patient_ids"]]

        train_df = df[df["patient_id"].isin(train_ids)]
        test_df  = df[df["patient_id"].isin(test_ids)]

        feature_cols      = [c for c in df.columns if c not in ("patient_id", "label")]
        X_train           = train_df[feature_cols].to_numpy(dtype=np.float32)
        y_train           = train_df["label"].to_numpy(dtype=np.int8)
        X_test            = test_df[feature_cols].to_numpy(dtype=np.float32)
        y_test            = test_df["label"].to_numpy(dtype=np.int8)
        test_patient_ids  = test_df["patient_id"].tolist()

        print(f"  Train: {len(X_train)} patients  |  Test: {len(X_test)} patients")
        print(f"  Test positives: {int(y_test.sum())}  negatives: {int((y_test==0).sum())}")

        # ── 2. Scale — fit on train only ───────────────────────────────────────
        scaler      = StandardScaler()
        X_train_s   = scaler.fit_transform(X_train)
        X_test_s    = scaler.transform(X_test)

        # ── 3. Resampling (training set only) ──────────────────────────────────
        pre_counts  = Counter(y_train.tolist())
        print(f"  Train class balance before resampling: {dict(pre_counts)}")

        if resampler is not None:
            X_train_r, y_train_r = resampler.fit_resample(X_train_s, y_train)
            post_counts = Counter(y_train_r.tolist())
            print(f"  Train class balance after  resampling: {dict(post_counts)}")
        else:
            X_train_r, y_train_r = X_train_s, y_train
            post_counts = pre_counts
            print("  No resampling. Using scale_pos_weight for imbalance.")

        # ── 4. XGBoost — scale_pos_weight only when no resampler ──────────────
        xgb_params = dict(XGBOOST_BASE_PARAMS)
        if resampler is None:
            n_neg = int((y_train_r == 0).sum())
            n_pos = int((y_train_r == 1).sum())
            xgb_params["scale_pos_weight"] = round(n_neg / n_pos, 4)
            print(f"  scale_pos_weight = {xgb_params['scale_pos_weight']:.4f}")

        model = XGBClassifier(**xgb_params)
        model.fit(X_train_r, y_train_r)

        # ── 5. Predict ─────────────────────────────────────────────────────────
        y_prob = model.predict_proba(X_test_s)[:, 1]
        y_pred = model.predict(X_test_s)

        # ── 6. Metrics ─────────────────────────────────────────────────────────
        f1_mac  = f1_score(y_test, y_pred, average="macro", zero_division=0)
        recall  = recall_score(y_test, y_pred, zero_division=0)
        prec    = precision_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else float("nan")
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        spec    = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        print(f"  f1_macro={f1_mac:.4f}  recall={recall:.4f}  "
              f"spec={spec:.4f}  precision={prec:.4f}  roc_auc={roc_auc:.4f}")

        # ── 7. Store fold result ───────────────────────────────────────────────
        output_data["fold_results"].append({
            "fold":              fold_id,
            "patient_ids":       test_patient_ids,
            "y_true":            y_test.tolist(),
            "y_prob":            y_prob.tolist(),
            "y_pred":            y_pred.tolist(),
            "train_class_balance_pre_resampling":  {
                str(k): int(v) for k, v in sorted(pre_counts.items())
            },
            "train_class_balance_post_resampling": {
                str(k): int(v) for k, v in sorted(post_counts.items())
            },
            "f1_macro":    round(float(f1_mac),  4),
            "recall":      round(float(recall),  4),
            "precision":   round(float(prec),    4),
            "specificity": round(float(spec),    4),
            "roc_auc":     round(float(roc_auc), 4) if not np.isnan(roc_auc) else None,
            "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        })

    # ── Cross-validation summary ───────────────────────────────────────────────
    metric_keys = ["f1_macro", "recall", "precision", "specificity", "roc_auc"]
    print()
    print("=" * 62)
    print(f"CROSS-VALIDATION SUMMARY  [resampler={resampler_str}]")
    print("=" * 62)
    for k in metric_keys:
        vals  = [f["roc_auc"] if k == "roc_auc" else f[k]
                 for f in output_data["fold_results"] if f.get(k) is not None]
        mean  = float(np.mean(vals)) if vals else 0.0
        std   = float(np.std(vals))  if vals else 0.0
        print(f"  {k:<14}: {mean:.4f} +/- {std:.4f}")

    # ── Write JSON ─────────────────────────────────────────────────────────────
    with open(str(output_file), "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)

    print()
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
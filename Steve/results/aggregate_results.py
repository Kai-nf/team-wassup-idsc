"""
aggregate_results.py
====================
Master ablation table generator for the Brugada Syndrome detection project.

Reads all result JSON files, computes patient-level metrics per fold, and
produces a ranked comparison table across every model tried.

Handles both JSON schemas used in this project:

  Schema A — "folds" dict (CNN, SVM, Ensemble outputs):
    data["folds"]["0"] = {y_true_patient, y_prob_patient, y_pred_patient, ...}

  Schema B — "fold_results" list (Method 4: LR, RF, XGBoost outputs):
    data["fold_results"][0] = {y_true, y_prob, y_pred, ...}

Usage:
    python aggregate_results.py

Output:
    results/master_ablation_table.csv
    results/master_ablation_table.tex
    Console: Markdown table + per-model fold-level breakdown
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# =============================================================================
# === MODEL REGISTRY — add/remove rows here to control what appears in table ==
# =============================================================================

REPO_ROOT   = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"

# Each entry: (display_name, filename, schema, group)
#
# schema options:
#   "folds_dict"        — data["folds"]["0"], keys: y_true_patient, y_prob_patient
#   "fold_results_list" — data["fold_results"][0], keys: y_true, y_prob
#
# group: used to add visual section separators in the printed table

MODEL_REGISTRY = [
    # ── Baseline methods ──────────────────────────────────────────────────────
    ("Method 1 — LR",             "method1_logistic_none.json",             "fold_results_list", "Baseline"),
    ("Method 1 — RF",             "method1_rf_none.json",                   "fold_results_list", "Baseline"),
    ("Method 2 — LR",             "method2_logistic_none.json",             "fold_results_list", "Baseline"),
    ("Method 2 — RF",             "method2_rf_none.json",                   "fold_results_list", "Baseline"),
    # ── CNN ───────────────────────────────────────────────────────────────────
    ("Method 3.2 — CNN patient",  "method3.2_1dcnn_patient_level.json",     "folds_dict",        "CNN"),
    ("Method 3.1 — CNN beat",     "method3.1_1dcnn_beat_level.json",        "folds_dict",        "CNN"),
    # ── SVM ───────────────────────────────────────────────────────────────────
    ("SVM — v3.2",                "svm_ablation_v3.2.json",                 "folds_dict",        "SVM"),
    ("SVM — v4",                  "svm_ablation_v4.json",                   "folds_dict",        "SVM"),
    # ── Method 4: tabular classifiers ─────────────────────────────────────────
    ("Method 4 — LR+SMOTE",       "method4_logistic_smote.json",            "fold_results_list", "Method4"),
    ("Method 4 — RF+SMOTE",       "method4_rf_smote.json",                  "fold_results_list", "Method4"),
    ("Method 4 — XGB+ADASYN",     "method4_xgboost_adasyn.json",            "fold_results_list", "Method4"),
    ("Method 4 — XGB+SMOTE",      "method4_xgboost_smote.json",             "fold_results_list", "Method4"),
    # ── Ensembles ─────────────────────────────────────────────────────────────
    ("Ensemble CNN+LR 0.65/0.35", "ensemble_cnn065_lr035.json",             "folds_dict",        "Ensemble"),
    ("Ensemble CNN+RF 0.65/0.35", "ensemble_cnn065_rf035.json",             "folds_dict",        "Ensemble"),
    ("Ensemble CNN+XGB 0.6/0.4",  "ensemble_cnn06_xgb04.json",              "folds_dict",        "Ensemble"),
    ("Ensemble CNN+LR equal",     "ensemble_cnn050_lr050.json",             "folds_dict",        "Ensemble"),
]

N_FOLDS    = 5
THRESHOLD  = 0.5   # applied to y_prob to compute classification metrics


# =============================================================================
# SCHEMA EXTRACTORS
# =============================================================================

def extract_folds_dict(data: dict) -> list:
    """
    Schema A: data["folds"]["0"], ["1"], ... ["4"]
    Per-fold keys: y_true_patient, y_prob_patient
    """
    folds_raw = data.get("folds", {})
    folds = []
    for i in range(N_FOLDS):
        fold = folds_raw.get(str(i))
        if fold is None:
            continue
        y_true = fold.get("y_true_patient")
        y_prob = fold.get("y_prob_patient")
        if y_true is None or y_prob is None:
            continue
        folds.append((np.array(y_true, dtype=np.int8),
                      np.array(y_prob, dtype=np.float32)))
    return folds


def extract_fold_results_list(data: dict) -> list:
    """
    Schema B: data["fold_results"][0], [1], ... [4]
    Per-fold keys: y_true, y_prob
    """
    fold_results = data.get("fold_results", [])
    folds = []
    for fold in fold_results:
        y_true = fold.get("y_true")
        y_prob = fold.get("y_prob")
        if y_true is None or y_prob is None:
            continue
        folds.append((np.array(y_true, dtype=np.int8),
                      np.array(y_prob, dtype=np.float32)))
    return folds


EXTRACTORS = {
    "folds_dict":        extract_folds_dict,
    "fold_results_list": extract_fold_results_list,
}


# =============================================================================
# METRICS
# =============================================================================

def compute_fold_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                         threshold: float = THRESHOLD) -> dict | None:
    """Compute all six metrics for one fold. Returns None if single-class."""
    if len(np.unique(y_true)) < 2:
        return None

    y_pred = (y_prob >= threshold).astype(int)

    macro_f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    roc_auc   = roc_auc_score(y_true, y_prob)
    pr_auc    = average_precision_score(y_true, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity    = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "macro_f1":    macro_f1,
        "precision":   precision,
        "recall":      recall,
        "specificity": specificity,
        "roc_auc":     roc_auc,
        "pr_auc":      pr_auc,
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }


def summarise(values: list) -> tuple[float, float]:
    """Return (mean, std) or (nan, nan) for empty list."""
    if not values:
        return float("nan"), float("nan")
    return float(np.mean(values)), float(np.std(values))


def fmt(mean: float, std: float, decimals: int = 4) -> str:
    if np.isnan(mean):
        return "N/A"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


# =============================================================================
# MAIN
# =============================================================================

def aggregate_results():
    print("=" * 72)
    print("BRUGADA DETECTION — MASTER ABLATION TABLE")
    print(f"Results directory : {RESULTS_DIR}")
    print(f"Threshold         : {THRESHOLD}")
    print("=" * 72)
    print()

    rows        = []
    detail_rows = []   # fold-level breakdown for CSV export
    skipped     = []

    current_group = None

    for display_name, filename, schema, group in MODEL_REGISTRY:
        file_path = RESULTS_DIR / filename
        if not file_path.exists():
            skipped.append((display_name, filename))
            continue

        with open(str(file_path), "r", encoding="utf-8") as f:
            data = json.load(f)

        extractor = EXTRACTORS.get(schema)
        if extractor is None:
            warnings.warn(f"Unknown schema '{schema}' for {display_name} — skipping.")
            continue

        fold_data = extractor(data)
        if not fold_data:
            skipped.append((display_name, filename + " (no fold data extracted)"))
            continue

        # ── Per-fold metrics ───────────────────────────────────────────────────
        metrics_per_fold = {k: [] for k in
                            ["macro_f1","precision","recall","specificity","roc_auc","pr_auc"]}
        agg_tp = agg_tn = agg_fp = agg_fn = 0

        for fold_idx, (y_true, y_prob) in enumerate(fold_data):
            m = compute_fold_metrics(y_true, y_prob, THRESHOLD)
            if m is None:
                continue
            for k in metrics_per_fold:
                metrics_per_fold[k].append(m[k])
            agg_tp += m["TP"]; agg_tn += m["TN"]
            agg_fp += m["FP"]; agg_fn += m["FN"]

            detail_rows.append({
                "model":       display_name,
                "fold":        fold_idx,
                "macro_f1":    round(m["macro_f1"],    4),
                "precision":   round(m["precision"],   4),
                "recall":      round(m["recall"],      4),
                "specificity": round(m["specificity"], 4),
                "roc_auc":     round(m["roc_auc"],     4),
                "pr_auc":      round(m["pr_auc"],      4),
                "TP": m["TP"], "TN": m["TN"], "FP": m["FP"], "FN": m["FN"],
            })

        # ── Aggregate confusion matrix specificity ─────────────────────────────
        agg_spec = agg_tn / (agg_tn + agg_fp) if (agg_tn + agg_fp) > 0 else float("nan")

        row = {"Model": display_name, "Group": group, "_file": filename}
        for k in metrics_per_fold:
            mean, std = summarise(metrics_per_fold[k])
            row[k + "_mean"] = round(mean, 4) if not np.isnan(mean) else float("nan")
            row[k + "_std"]  = round(std,  4) if not np.isnan(std)  else float("nan")

        row["agg_TP"] = agg_tp; row["agg_TN"] = agg_tn
        row["agg_FP"] = agg_fp; row["agg_FN"] = agg_fn
        row["agg_specificity"] = round(agg_spec, 4)
        rows.append(row)

    # ── Print Markdown table ───────────────────────────────────────────────────
    COL = {
        "macro_f1":    "Macro F1",
        "precision":   "Precision",
        "recall":      "Recall",
        "specificity": "Specificity",
        "roc_auc":     "ROC AUC",
        "pr_auc":      "PR AUC",
    }

    W_MODEL = 30
    W_COL   = 18

    header = f"| {'Model':<{W_MODEL}} |" + "".join(
        f" {v:<{W_COL}} |" for v in COL.values()
    )
    divider = f"| {'-'*W_MODEL} |" + "".join(f" {'-'*W_COL} |" for _ in COL)

    current_group = None
    print(header)
    print(divider)

    for row in rows:
        if row["Group"] != current_group:
            current_group = row["Group"]
            print(f"| {'— ' + current_group + ' —':<{W_MODEL}} |" +
                  "".join(f" {'':^{W_COL}} |" for _ in COL))

        cells = []
        for k in COL:
            mean = row.get(k + "_mean", float("nan"))
            std  = row.get(k + "_std",  float("nan"))
            cells.append(fmt(mean, std, decimals=4))

        print(f"| {row['Model']:<{W_MODEL}} |" +
              "".join(f" {c:<{W_COL}} |" for c in cells))

    print()

    # ── Clinical targets check ────────────────────────────────────────────────
    print("=== CLINICAL TARGETS (recall >= 0.80, specificity >= 0.70) ===")
    print()
    for row in rows:
        rec  = row.get("recall_mean", float("nan"))
        spec = row.get("specificity_mean", float("nan"))
        if np.isnan(rec) or np.isnan(spec):
            continue
        r_ok = rec  >= 0.80
        s_ok = spec >= 0.70
        if r_ok and s_ok:
            status = "BOTH MET"
        elif r_ok:
            status = "recall OK"
        elif s_ok:
            status = "spec OK"
        else:
            status = "—"
        print(f"  {row['Model']:<32}  recall={rec:.4f}  spec={spec:.4f}  [{status}]")

    print()

    # ── Skipped files ─────────────────────────────────────────────────────────
    if skipped:
        print("=== SKIPPED (file not found) ===")
        for name, f in skipped:
            print(f"  {name}: {f}")
        print()

    # ── Save outputs ──────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Summary CSV ───────────────────────────────────────────────────────────
    summary_cols = ["Model", "Group"] + \
                   [f"{k}_mean" for k in COL] + \
                   [f"{k}_std"  for k in COL] + \
                   ["agg_TP", "agg_TN", "agg_FP", "agg_FN", "agg_specificity"]
    df_summary = pd.DataFrame(rows)
    df_summary = df_summary[[c for c in summary_cols if c in df_summary.columns]]
    csv_path = RESULTS_DIR / "master_ablation_table.csv"
    df_summary.to_csv(str(csv_path), index=False)
    print(f"Summary CSV saved  : {csv_path}")

    # ── Fold-level CSV ────────────────────────────────────────────────────────
    df_detail = pd.DataFrame(detail_rows)
    detail_path = RESULTS_DIR / "master_ablation_table_foldlevel.csv"
    df_detail.to_csv(str(detail_path), index=False)
    print(f"Fold-level CSV saved : {detail_path}")

    # ── LaTeX ─────────────────────────────────────────────────────────────────
    # Build a clean display-only DataFrame for LaTeX
    display_rows = []
    for row in rows:
        display_rows.append({
            "Model":       row["Model"],
            "Macro F1":    fmt(row.get("macro_f1_mean", float("nan")),
                               row.get("macro_f1_std",  float("nan"))),
            "Recall":      fmt(row.get("recall_mean",    float("nan")),
                               row.get("recall_std",     float("nan"))),
            "Specificity": fmt(row.get("specificity_mean", float("nan")),
                               row.get("specificity_std",  float("nan"))),
            "ROC AUC":     fmt(row.get("roc_auc_mean",  float("nan")),
                               row.get("roc_auc_std",   float("nan"))),
            "PR AUC":      fmt(row.get("pr_auc_mean",   float("nan")),
                               row.get("pr_auc_std",    float("nan"))),
        })

    df_latex = pd.DataFrame(display_rows)
    tex_path = RESULTS_DIR / "master_ablation_table.tex"
    df_latex.to_latex(str(tex_path), index=False, escape=True)
    print(f"LaTeX table saved    : {tex_path}")


if __name__ == "__main__":
    aggregate_results()
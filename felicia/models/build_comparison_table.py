# pyre-ignore-all-errors
import json
from pathlib import Path

import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "felicia" / "results" / "evaluation_metrics"

CANONICAL_OUTPUTS = {
    "method1_logistic.json",
    "method1_rf.json",
    "method2_logistic.json",
    "method2_rf.json",
    "method3_2_logistic.json",
    "method3_2_rf.json",
    "method4_logistic_smote.json",
    "method4_logistic_borderline_smote.json",
    "method4_logistic_adasyn.json",
    "method4_rf_smote.json",
    "method4_rf_borderline_smote.json",
    "method4_rf_adasyn.json",
}


def _format_metric(mean, std, decimals=3):
    """Format metric as 'mean±std'."""
    if mean is None or std is None:
        return "—"
    return f"{mean:.{decimals}f}±{std:.{decimals}f}"


def _read_summary(path: Path):
    """Extract Phase 5 metrics summary from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    summary = data.get("summary", {})
    agg_cm = data.get("aggregated_confusion_matrix", {})
    fold_results = data.get("fold_results", [])

    pos_counts = []
    for r in fold_results:
        if "n_positive" in r:
            pos_counts.append(int(r["n_positive"]))
        elif "n_positive_test" in r:
            pos_counts.append(int(r["n_positive_test"]))
    
    # Extract all Phase 5 metrics
    row = {
        "file": path.name,
        "method": data.get("method"),
        "model": data.get("model"),
        "resampler": data.get("resampler", "none"),
        
        # Phase 5 metrics (mean ± std)
        "f1_macro": _format_metric(
            summary.get("f1_macro", {}).get("mean"),
            summary.get("f1_macro", {}).get("std")
        ),
        "precision_brugada": _format_metric(
            summary.get("precision_brugada", {}).get("mean"),
            summary.get("precision_brugada", {}).get("std")
        ),
        "recall_brugada": _format_metric(
            summary.get("recall_brugada", {}).get("mean"),
            summary.get("recall_brugada", {}).get("std")
        ),
        "f1_brugada": _format_metric(
            summary.get("f1_brugada", {}).get("mean"),
            summary.get("f1_brugada", {}).get("std")
        ),
        "specificity": _format_metric(
            summary.get("specificity", {}).get("mean"),
            summary.get("specificity", {}).get("std")
        ),
        "roc_auc": _format_metric(
            summary.get("roc_auc", {}).get("mean"),
            summary.get("roc_auc", {}).get("std")
        ),
        "pr_auc": _format_metric(
            summary.get("pr_auc", {}).get("mean"),
            summary.get("pr_auc", {}).get("std")
        ),
        "sens_at_95spec": _format_metric(
            summary.get("sens_at_95spec", {}).get("mean"),
            summary.get("sens_at_95spec", {}).get("std")
        ),
        
        # Aggregated confusion matrix
        "cm_tp": agg_cm.get("TP"),
        "cm_tn": agg_cm.get("TN"),
        "cm_fp": agg_cm.get("FP"),
        "cm_fn": agg_cm.get("FN"),
        "n_positive_per_fold": ",".join(str(v) for v in pos_counts) if pos_counts else "—",
        "n_positive_mean": f"{np.mean(pos_counts):.1f}" if pos_counts else "—",
        "n_positive_std": f"{np.std(pos_counts):.2f}" if pos_counts else "—",
        "n_positive_total": int(sum(pos_counts)) if pos_counts else None,
    }
    
    # Compute overall statistics from aggregated CM
    if row["cm_tp"] is not None and row["cm_tn"] is not None and row["cm_fp"] is not None and row["cm_fn"] is not None:
        tp, tn, fp, fn = int(row["cm_tp"]), int(row["cm_tn"]), int(row["cm_fp"]), int(row["cm_fn"])
        overall_sen: float | None = tp / (tp + fn) if (tp + fn) > 0 else None
        overall_spec: float | None = tn / (tn + fp) if (tn + fp) > 0 else None
        row["agg_cm_sensitivity"] = f"{overall_sen:.3f}" if overall_sen is not None else "—"
        row["agg_cm_specificity"] = f"{overall_spec:.3f}" if overall_spec is not None else "—"
    else:
        row["agg_cm_sensitivity"] = "—"
        row["agg_cm_specificity"] = "—"

    if row["cm_tp"] is not None and row["cm_tn"] is not None and row["cm_fp"] is not None and row["cm_fn"] is not None:
        row["agg_cm_counts"] = f"TP={row['cm_tp']}, TN={row['cm_tn']}, FP={row['cm_fp']}, FN={row['cm_fn']}"
    else:
        row["agg_cm_counts"] = "—"
    
    return row


def main():
    paths = sorted(
        p
        for p in RESULT_DIR.glob("method*.json")
        if p.name in CANONICAL_OUTPUTS
    )
    if not paths:
        raise FileNotFoundError(f"No method result JSON files found in {RESULT_DIR}")

    rows = [_read_summary(p) for p in paths]
    
    # Create main comparison table (exclude raw CM values)
    display_cols = [
        "method", "model", "resampler",
        "f1_macro", "f1_brugada", "precision_brugada", "recall_brugada", "specificity",
        "roc_auc", "pr_auc", "sens_at_95spec",
        "n_positive_total", "n_positive_mean", "n_positive_std", "n_positive_per_fold",
        "agg_cm_sensitivity", "agg_cm_specificity", "agg_cm_counts"
    ]
    df_display = pd.DataFrame(rows)[display_cols]
    df_display = df_display.sort_values(["method", "model", "resampler"], na_position="last")
    
    # Create detailed table with raw CM values (for reference)
    all_cols = list(df_display.columns) + ["cm_tp", "cm_tn", "cm_fp", "cm_fn"]
    df_detailed = pd.DataFrame(rows)[all_cols]
    df_detailed = df_detailed.sort_values(["method", "model", "resampler"], na_position="last")
    
    csv_path = RESULT_DIR / "phase5_comparison_summary.csv"
    md_path = RESULT_DIR / "phase5_comparison_summary.md"
    csv_detailed_path = RESULT_DIR / "phase5_comparison_detailed.csv"
    
    # Save display version
    df_display.to_csv(csv_path, index=False)
    md_lines = [
        "# Phase 5 Metrics Comparison Summary",
        "",
        "**Metrics Format**: `mean±std` across 5 folds",
        "",
        "**Columns**:",
        "- `f1_macro`: Macro-averaged F1 (class-balanced)",
        "- `f1_brugada`: Binary F1 score for Brugada class (pos_label=1)",
        "- `precision_brugada`: Positive Predictive Value (PPV) for Brugada class",
        "- `recall_brugada`: Sensitivity / True Positive Rate for Brugada class",
        "- `specificity`: True Negative Rate (rule-out confidence)",
        "- `roc_auc`: Threshold-independent discrimination (ROC curve AUC)",
        "- `pr_auc`: Importance of Brugada positives (PR curve AUC, imbalance-sensitive)",
        "- `sens_at_95spec`: Sensitivity when specificity held at ≥95%",
        "- `n_positive_total`: Total Brugada-positive test samples across all 5 folds",
        "- `n_positive_mean`, `n_positive_std`: Mean ± std Brugada-positive count per test fold",
        "- `n_positive_per_fold`: Per-fold positive counts (context for PR-AUC variability)",
        "- `agg_cm_sensitivity`: Overall sensitivity from aggregated confusion matrix across all folds",
        "- `agg_cm_specificity`: Overall specificity from aggregated confusion matrix across all folds",
        "- `agg_cm_counts`: Summed TP/TN/FP/FN across all 5 patient-disjoint test folds",
        "",
        df_display.to_markdown(index=False),
    ]
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    
    # Save detailed version (with raw CM values)
    df_detailed.to_csv(csv_detailed_path, index=False)
    
    print(f"✓ Saved {csv_path}")
    print(f"✓ Saved {md_path}")
    print(f"✓ Saved {csv_detailed_path} (detailed with raw CM values)")
    print(f"\n{len(df_display)} experiments loaded and compared")


if __name__ == "__main__":
    main()
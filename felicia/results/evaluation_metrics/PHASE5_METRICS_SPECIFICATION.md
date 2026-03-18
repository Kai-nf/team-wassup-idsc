# Phase 5 Metrics Specification (Integrated into Phase 4)

## Scope

This specification defines how all Phase 5 metrics are computed and reported for each experiment.

- **Outer evaluation**: 5-fold patient-level disjoint cross-validation (`master_folds_drop14.json`)
- **Reporting unit**: mean ± std across folds (except aggregated confusion matrix)
- **Positive class**: Brugada (label = 1)

## Core Rules

1. Metrics are computed **per fold** on that fold's test set, then aggregated as mean ± std.
2. Threshold-independent metrics (ROC-AUC, PR-AUC) are **never** computed from aggregated fold data.
3. Sensitivity at 95% specificity is computed per fold from the fold ROC curve (no global threshold).
4. Confusion matrix counts are summed across folds only because each patient appears in exactly one test fold.
5. `n_positive` / `n_negative` are tracked per fold to contextualize PR-AUC variability.

## Metric Definitions

| Metric | Formula / Function | Notes |
|--------|-------------------|-------|
| **F1 (Macro)** | `f1_score(y_true, y_pred, average="macro")` | Balanced across both classes |
| **Precision (Brugada)** | `precision_score(..., pos_label=1)` = TP / (TP+FP) | PPV for Brugada class |
| **Recall / Sensitivity (Brugada)** | `recall_score(..., pos_label=1)` = TP / (TP+FN) | Most critical clinical metric |
| **Specificity** | TN / (TN+FP) from fold confusion matrix | True Negative Rate |
| **ROC-AUC** | `roc_auc_score(y_true, y_prob)` | Threshold-independent; uses probabilities |
| **PR-AUC** | `average_precision_score(y_true, y_prob, pos_label=1)` | Imbalance-sensitive; uses probabilities |
| **Sens @ 95% Spec** | Sensitivity at threshold where specificity ≥ 95% | Per-fold ROC curve search |

### Sensitivity @ 95% Specificity — Procedure

1. Compute ROC curve: `fpr, tpr, thresholds = roc_curve(y_true, y_prob)`
2. Find indices where `specificity = 1 - fpr >= 0.95`
3. Among eligible indices, pick the one with highest `tpr`
4. Record that sensitivity for the fold (fallback: sensitivity at highest achievable specificity)

### Aggregated Confusion Matrix

- **Construction**: Sum TP, TN, FP, FN across all 5 test folds
- **Validity**: Valid because test folds are patient-exclusive (no sample appears twice)
- **Derived**: Overall Sensitivity = Σ(TP) / Σ(TP+FN); Overall Specificity = Σ(TN) / Σ(TN+FP)

## Per-Fold Sample Counts

**Tracked fields** (required per fold):
- `n_positive`: Count of Brugada samples in test fold
- `n_negative`: Count of non-Brugada samples in test fold
- Backward-compatible aliases: `n_positive_test`, `n_negative_test`

**Why important**: PR-AUC is sensitive to class prevalence within each fold. Reporting
fold-level positive counts explains why PR-AUC std may be higher in folds with fewer positives.

### Expected counts per fold (this cohort, 349 patients, ~27% Brugada)

| Fold | n_positive | n_negative |
|------|-----------|-----------|
| 0    | 15        | 55        |
| 1    | 15        | 55        |
| 2    | 14        | 56        |
| 3    | 14        | 56        |
| 4    | 14        | 55        |

## Output Contract

Each experiment JSON includes:
- `summary` with `{mean, std}` for all fold-level metrics
- `aggregated_confusion_matrix` with summed counts (`TP`, `TN`, `FP`, `FN`)
- `fold_results[*].n_positive` and `fold_results[*].n_negative` (and backward-compat aliases)
- `fold_test_positive_counts` and `fold_test_negative_counts` at top level

### JSON Schema (summary section)

```json
{
  "fold_test_positive_counts": [15, 15, 14, 14, 14],
  "fold_test_negative_counts": [55, 55, 56, 56, 55],
  "fold_results": [
    {
      "fold": 0,
      "f1_macro": 0.638, "precision_brugada": 0.4,
      "recall_brugada": 0.533, "specificity": 0.782,
      "roc_auc": 0.714, "pr_auc": 0.401, "sens_at_95spec": 0.133,
      "n_positive": 15, "n_negative": 55,
      "n_positive_test": 15, "n_negative_test": 55,
      "confusion_matrix": [[TN, FP], [FN, TP]], ...
    }, ...
  ],
  "aggregated_confusion_matrix": {
    "TP": 30, "TN": 213, "FP": 64, "FN": 42, ...
  },
  "summary": {
    "f1_macro":            {"mean": 0.578, "std": 0.072},
    "precision_brugada":   {"mean": 0.315, "std": 0.088},
    "recall_brugada":      {"mean": 0.417, "std": 0.160},
    "specificity":         {"mean": 0.769, "std": 0.054},
    "roc_auc":             {"mean": 0.651, "std": 0.097},
    "pr_auc":              {"mean": 0.374, "std": 0.107},
    "sens_at_95spec":      {"mean": 0.155, "std": 0.146}
  }
}
```

## Reporting Table Columns

| Column | Description |
|--------|-------------|
| `method` | Preprocessing method (method1–method4) |
| `model` | Classifier (logistic / rf) |
| `resampler` | Resampling strategy (method4 only) |
| `f1_macro` | Macro F1 mean±std |
| `precision_brugada` | Brugada PPV mean±std |
| `recall_brugada` | Brugada sensitivity mean±std |
| `specificity` | Specificity mean±std |
| `roc_auc` | ROC-AUC mean±std |
| `pr_auc` | PR-AUC mean±std |
| `sens_at_95spec` | Sensitivity at ≥95% specificity, mean±std |
| `n_positive_total` | Total positives across all 5 test folds |
| `n_positive_mean` | Mean positives per fold |
| `n_positive_std` | Std of positives per fold |
| `n_positive_per_fold` | Per-fold positive counts (comma-separated) |
| `agg_cm_sensitivity` | Sensitivity from aggregated CM |
| `agg_cm_specificity` | Specificity from aggregated CM |
| `agg_cm_counts` | Raw TP/TN/FP/FN from aggregated CM |

## Clinical Interpretation Guide

| Metric | Ideal | Interpretation |
|--------|-------|----------------|
| Recall (Sensitivity) | > 0.85 | Catch >85% of true Brugada patients |
| Specificity | > 0.90 | Correctly rule out >90% of non-Brugada |
| Precision (PPV) | > 0.50 | >50% confidence when predicting Brugada |
| ROC-AUC | > 0.70 | Discriminates better than random |
| PR-AUC | > 0.40 | Prioritizes Brugada well vs. baseline |
| Sens@95%Spec | > 0.20 | Detect >20% of Brugada at high specificity |

## Changelog

- **2026-03-17**: Expanded specification; locked n_positive per fold schema; updated table columns
- **2025-03-17**: Initial specification, Phase 5 metrics locked, per-fold sample counts added
- **Revision Status**: Final, ready for publication

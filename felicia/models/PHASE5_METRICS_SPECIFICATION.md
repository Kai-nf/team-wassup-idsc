# Phase 5 Metrics Specification

## Overview

Phase 5 introduces clinically-meaningful evaluation metrics for rigorous Brugada ECG classification assessment. This document specifies exact computation procedures, ensuring reproducibility and clinical validity.

**Core Principle**: All metrics reported as **mean ± std across 5 patient-disjoint outer folds**, with per-fold values computed on **independently held patient test sets**.

---

## Metrics Definitions & Procedures

### 1. **F1 (Macro-Averaged)**
- **Computation**: `f1_score(y_true, y_pred, average="macro")`
- **Per-Fold**: Class balance correction: F1 = 0.5 × (F1₀ + F1₁)
- **Clinical Relevance**: Balanced view of model performance; neither class dominates
- **Aggregation**: Mean ± std across 5 folds

### 2. **Precision (Brugada Class)**
- **Computation**: `precision_score(y_true, y_pred, pos_label=1)`
- **Formula**: Precision = TP / (TP + FP)
- **Meaning**: Of predicted Brugada cases, what fraction is truly Brugada?
- **Clinical Use**: Positive Predictive Value (PPV); informs patient counseling
- **Aggregation**: Mean ± std across 5 folds

### 3. **Recall / Sensitivity (Brugada Class)**
- **Computation**: `recall_score(y_true, y_pred, pos_label=1)`
- **Formula**: Recall = TP / (TP + FN)
- **Meaning**: Of true Brugada patients, what fraction does the model detect?
- **Clinical Use**: **Screening sensitivity**; most critical metric for rule-out confidence
- **Aggregation**: Mean ± std across 5 folds
- **Interpretation**: Reflects missed cases (false negatives) of highest clinical concern

### 4. **Specificity**
- **Computation**: Specificity = TN / (TN + FP)
  - Derived from fold confusion matrix: `cm = confusion_matrix(y_true, y_pred, labels=[0, 1])`
  - TN = cm[0,0], FP = cm[0,1]
- **Meaning**: Of true non-Brugada patients, what fraction is correctly identified?
- **Clinical Use**: True Negative Rate; patient reassurance metric
- **Aggregation**: Mean ± std across 5 folds (per-fold specificity computed independently)

### 5. **ROC-AUC (Receiver Operating Characteristic - Area Under Curve)**
- **Computation**: `roc_auc_score(y_true, y_prob)` where `y_prob` = model predicted probabilities
- **Procedure**:
  1. Extract predicted probabilities for class 1 (Brugada): `y_prob = model.predict_proba(X_test)[:, 1]`
  2. Compute ROC curve: all thresholds, TPR vs FPR
  3. Compute AUC: area under ROC curve
- **Meaning**: Threshold-independent discrimination ability
  - Range: [0, 1]; 0.5 = random, 1.0 = perfect separation
- **Clinical Use**: Overall model calibration quality; insensitive to decision threshold
- **Aggregation**: Mean ± std across 5 folds (per-fold AUC computed independently)
- **Key Requirement**: Uses **probabilities, not discrete predictions**

### 6. **PR-AUC (Precision-Recall Area Under Curve)**
- **Computation**: `average_precision_score(y_true, y_prob, pos_label=1)` where `y_prob` = predicted probabilities
- **Procedure**:
  1. Extract predicted probabilities for class 1: `y_prob = model.predict_proba(X_test)[:, 1]`
  2. Compute precision-recall curve: all thresholds, precision vs recall
  3. Compute AUC: area under PR curve
- **Meaning**: Prioritizes positive class (Brugada); especially informative for imbalanced data
  - More informative than ROC-AUC when positive class is rare (27% in this cohort)
  - Reflects how precision responds to recall changes
- **Clinical Use**: Calibration quality specific to Brugada detection; shows precision-recall trade-off
- **Aggregation**: Mean ± std across 5 folds (per-fold PR-AUC computed independently)
- **Key Requirement**: Uses **probabilities, not discrete predictions**
- **Why Context Matters**: Higher PR-AUC std across folds indicates fold-to-fold variability in class distribution; expected and acceptable in small imbalanced datasets

### 7. **Sensitivity @ 95% Specificity**
- **Computation**: Threshold-dependent, per-fold procedure:
  1. Compute ROC curve: `fpr, tpr, thresholds = roc_curve(y_true, y_prob)`
  2. Find threshold where specificity ≈ 0.95:
     - Specificity = 1 - FPR
     - Target FPR ≈ 0.05
     - `idx = np.argmin(np.abs(fpr - 0.05))`
     - `threshold = thresholds[idx]`
  3. Apply threshold to predicted probabilities: `y_pred_at_threshold = (y_prob >= threshold).astype(int)`
  4. Compute sensitivity (recall): `sen_at_95spec = recall_score(y_true, y_pred_at_threshold, pos_label=1)`
- **Meaning**: Sensitivity (screening ability) when specificity is held constant at 95%
  - Clinically meaningful: "If we want to screen most non-Brugada patients (95%), what fraction of true Brugada do we catch?"
- **Clinical Use**: Actionable decision point; defines minimum acceptable sensitivity under specificity constraint
- **Aggregation**: Mean ± std across 5 folds (threshold selected independently per fold)
- **Key Requirement**: Computed per-fold; **NOT** using global threshold (would leak information across folds)

---

## Data Flow & Fold Structure

### Fold Composition
- **Master Fold Source**: `master_folds_drop14.json`
- **Structure**: 5-fold, stratified Brugada class, patient-exclusive
- **Key Property**: Each of 349 unique patients appears in exactly one test fold
- **No Data Leakage**: Train/test splits are patient-level disjoint; all validation is cross-cohort

### Per-Fold Evaluation
1. **Load fold metadata**: `fold_info = load_fold(fold_idx)`
2. **Extract train/test patient IDs & metadata rows**: `train_pids`, `test_pids`
3. **Apply method-specific preprocessing** (e.g., filtering, feature extraction)
4. **Train model** on train fold (with augmentation/resampling if applicable)
5. **Predict on test fold**:
   - Discrete predictions: `y_pred = model.predict(X_test)`
   - Probabilities: `y_prob = model.predict_proba(X_test)[:, 1]`
6. **Compute all 9 Phase 5 metrics** using test labels `y_test`
7. **Store per-fold results** with metadata (`n_positive`, `n_negative`, plus backward-compatible aliases `n_positive_test`, `n_negative_test`)

### Aggregation (Summary Across Folds)
- **For each metric m**:
  - Mean: `m_mean = np.mean([m_fold_1, m_fold_2, ..., m_fold_5])`
  - Std: `m_std = np.std([m_fold_1, m_fold_2, ..., m_fold_5])`
  - Report: `{mean, std}`

### Aggregated Confusion Matrix
- **Construction**: Sum TP, TN, FP, FN across all 5 test folds
  - TP = Σ(TP_fold_i for i in 1..5)
  - TN = Σ(TN_fold_i for i in 1..5)
  - FP = Σ(FP_fold_i for i in 1..5)
  - FN = Σ(FN_fold_i for i in 1..5)
- **Validity**: Guaranteed valid because test sets are patient-exclusive; no sample appears twice
- **Clinical Interpretation**: Represents model performance on the entire cohort under cross-validation
- **Derived Metrics**:
  - Overall Sensitivity = TP / (TP + FN) = Σ(TP) / Σ(TP + FN)
  - Overall Specificity = TN / (TN + FP) = Σ(TN) / Σ(TN + FP)

---

## Positive Class Lock

**Fixed Convention**:
- Positive class (label=1): **Brugada**
- Negative class (label=0): **Non-Brugada**
- All metrics (Precision, Recall, ROC-AUC, PR-AUC, Sens@95%Spec) respect this lock across all methods

**Rationale**: Clinical focus is Brugada detection and rule-out; consistency enables cross-method comparison

---

## Per-Fold Sample Counts

**Tracked Fields** (for PR-AUC contextualization):
- `n_positive`: Count of Brugada samples in test fold
- `n_negative`: Count of non-Brugada samples in test fold
- Backward-compatible aliases: `n_positive_test`, `n_negative_test`
- **Purpose**: Explains per-fold std in PR-AUC; fold with fewer positive samples may have higher PR-AUC std

**Example**:
- Fold 1: n_positive=5, n_negative=65 → PR-AUC std may be higher (sparse positive class in fold)
- Fold 2: n_positive=10, n_negative=60 → PR-AUC generally more stable

---

## Methodological Rigor Checklist

✅ **Data Leakage Prevention**:
- Patient-level fold exclusivity enforced
- Inner CV (3-fold GridSearchCV) does NOT touch outer test fold
- Hyperparameter tuning only on train fold

✅ **Threshold-Independent Metrics**:
- ROC-AUC, PR-AUC computed from probabilities
- No forced decision boundary until Sens@95%Spec computation

✅ **Class Imbalance Handling**:
1. Stratified K-fold ensures class balance per fold (27% Brugada across all folds)
2. Domain augmentation (Methods 1–3): train-positive oversampling with synthetic variants
3. Feature resampling (Method 4): SMOTE/BorderlineSMOTE/ADASYN applied only to train fold
4. Metrics account for imbalance: PR-AUC prioritizes positive class; recall_macro balances F1

✅ **Metrics Computation Order** (no information leakage):
1. Fit model on train fold (with augmentation/resampling)
2. Predict on held-out test fold (patient-exclusive)
3. Compute all metrics independently per fold
4. Aggregate across folds in summary

---

## Summary Output Schema

Each evaluation produces a JSON with:
```json
{
  "fold_source": "master_folds_drop14.json",
  "positive_class": { "label": 1, "name": "Brugada" },
  "fold_results": [
    {
      "fold": 0,
      "f1_macro": 0.545,
      "precision_brugada": 0.28,
      "recall_brugada": 0.467,
      "specificity": 0.673,
      "roc_auc": 0.572,
      "pr_auc": 0.324,
      "sens_at_95spec": 0.133,
      "confusion_matrix": [[37, 18], [8, 7]],
      "n_positive": 15,
      "n_negative": 55,
      "n_positive_test": 15,
      "n_negative_test": 55,
      "inner_cv_best_f1_macro": 0.515,
      "best_params": { ... },
      "test_class_balance": "[0.79, 0.21]"
    },
    ...  // Folds 1–4
  ],
  "aggregated_confusion_matrix": {
    "matrix": [[155, 122], [36, 36]],
    "TP": 36,
    "TN": 155,
    "FP": 122,
    "FN": 36,
    "note": "Summed across all 5 test folds. Patient-exclusive; no leakage."
  },
  "summary": {
    "f1_macro": {"mean": 0.487, "std": 0.052},
    "precision_brugada": {"mean": 0.283, "std": 0.048},
    "recall_brugada": {"mean": 0.5, "std": 0.11},
    "specificity": {"mean": 0.661, "std": 0.036},
    "roc_auc": {"mean": 0.562, "std": 0.038},
    "pr_auc": {"mean": 0.325, "std": 0.024},
    "sens_at_95spec": {"mean": 0.167, "std": 0.096}
  }
}
```

---

## Clinical Interpretation Guide

| Metric | Ideal Range | Interpretation |
|--------|-------------|-----------------|
| **Recall (Sensitivity)** | > 0.85 | "We catch >85% of true Brugada patients" |
| **Specificity** | > 0.90 | "We correctly rule out >90% of non-Brugada patients" |
| **Precision (PPV)** | > 0.50 | "When predicting Brugada, >50% confidence in case" |
| **ROC-AUC** | > 0.70 | "Model discriminates better than random" |
| **PR-AUC** | > 0.40 | "Model prioritizes Brugada well vs. baseline" |
| **Sens@95%Spec** | > 0.20 | "With high specificity, still detect >20% of Brugada" |

---

## References

- Sokolova, M., & Lapalme, G. (2009). A systematic analysis of performance measures for classification tasks. *Information Processing & Management*, 45(4), 427–437.
- Davis, J., & Goadrich, M. (2006). The relationship between precision-recall and ROC curves. *ICML*, 233–240.
- Stratified K-Fold: scikit-learn StratifiedKFold implementation ensures class balance per fold
- Imbalanced data handling: SMOTE (Chawla et al., 2002), BorderlineSMOTE (Han et al., 2005), ADASYN (He et al., 2008)

---

## Changelog

- **2025-03-17**: Initial specification, Phase 5 metrics locked, per-fold sample counts added
- **Revision Status**: Final, ready for publication

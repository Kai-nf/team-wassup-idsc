# Phase 5 Metrics Comparison Summary

**Metrics Format**: `mean±std` across 5 folds

**Columns**:
- `f1_macro`: Macro-averaged F1 (class-balanced)
- `f1_brugada`: Binary F1 score for Brugada class (pos_label=1)
- `precision_brugada`: Positive Predictive Value (PPV) for Brugada class
- `recall_brugada`: Sensitivity / True Positive Rate for Brugada class
- `specificity`: True Negative Rate (rule-out confidence)
- `roc_auc`: Threshold-independent discrimination (ROC curve AUC)
- `pr_auc`: Importance of Brugada positives (PR curve AUC, imbalance-sensitive)
- `sens_at_95spec`: Sensitivity when specificity held at ≥95%
- `n_positive_total`: Total Brugada-positive test samples across all 5 folds
- `n_positive_mean`, `n_positive_std`: Mean ± std Brugada-positive count per test fold
- `n_positive_per_fold`: Per-fold positive counts (context for PR-AUC variability)
- `agg_cm_sensitivity`: Overall sensitivity from aggregated confusion matrix across all folds
- `agg_cm_specificity`: Overall specificity from aggregated confusion matrix across all folds
- `agg_cm_counts`: Summed TP/TN/FP/FN across all 5 patient-disjoint test folds

| method    | model    | resampler        | f1_macro    | f1_brugada   | precision_brugada   | recall_brugada   | specificity   | roc_auc     | pr_auc      | sens_at_95spec   |   n_positive_total |   n_positive_mean |   n_positive_std | n_positive_per_fold   |   agg_cm_sensitivity |   agg_cm_specificity | agg_cm_counts                |
|:----------|:---------|:-----------------|:------------|:-------------|:--------------------|:-----------------|:--------------|:------------|:------------|:-----------------|-------------------:|------------------:|-----------------:|:----------------------|---------------------:|---------------------:|:-----------------------------|
| method1   | logistic | none             | 0.481±0.031 | 0.268±0.078  | 0.209±0.039         | 0.404±0.170      | 0.622±0.120   | 0.544±0.050 | 0.248±0.031 | 0.014±0.029      |                 72 |              14.4 |             0.49 | 15,15,14,14,14        |                0.403 |                0.621 | TP=29, TN=172, FP=105, FN=43 |
| method1   | rf       | none             | 0.457±0.029 | 0.027±0.053  | 0.200±0.400         | 0.014±0.029      | 1.000±0.000   | 0.529±0.044 | 0.262±0.046 | 0.028±0.034      |                 72 |              14.4 |             0.49 | 15,15,14,14,14        |                0.014 |                1     | TP=1, TN=277, FP=0, FN=71    |
| method2   | logistic | none             | 0.523±0.031 | 0.268±0.046  | 0.256±0.038         | 0.304±0.120      | 0.761±0.119   | 0.524±0.070 | 0.281±0.064 | 0.055±0.068      |                 72 |              14.4 |             0.49 | 15,15,14,14,14        |                0.306 |                0.762 | TP=22, TN=211, FP=66, FN=50  |
| method2   | rf       | none             | 0.442±0.002 | 0.000±0.000  | 0.000±0.000         | 0.000±0.000      | 1.000±0.000   | 0.586±0.051 | 0.286±0.051 | 0.082±0.049      |                 72 |              14.4 |             0.49 | 15,15,14,14,14        |                0     |                1     | TP=0, TN=277, FP=0, FN=72    |
| method3_2 | logistic | none             | 0.580±0.061 | 0.377±0.094  | 0.313±0.075         | 0.474±0.125      | 0.733±0.026   | 0.635±0.084 | 0.380±0.113 | 0.140±0.126      |                 72 |              14.4 |             0.49 | 15,15,14,14,14        |                0.472 |                0.733 | TP=34, TN=203, FP=74, FN=38  |
| method3_2 | rf       | none             | 0.543±0.070 | 0.204±0.134  | 0.433±0.276         | 0.138±0.096      | 0.964±0.023   | 0.709±0.087 | 0.404±0.088 | 0.169±0.075      |                 72 |              14.4 |             0.49 | 15,15,14,14,14        |                0.139 |                0.964 | TP=10, TN=267, FP=10, FN=62  |
| method4   | logistic | adasyn           | 0.670±0.053 | 0.520±0.073  | 0.424±0.071         | 0.682±0.101      | 0.755±0.047   | 0.817±0.039 | 0.590±0.075 | 0.388±0.043      |                 72 |              14.4 |             0.49 | 15,15,14,14,14        |                0.681 |                0.755 | TP=49, TN=209, FP=68, FN=23  |
| method4   | logistic | borderline_smote | 0.708±0.087 | 0.565±0.127  | 0.489±0.141         | 0.681±0.127      | 0.805±0.064   | 0.831±0.055 | 0.611±0.107 | 0.443±0.094      |                 72 |              14.4 |             0.49 | 15,15,14,14,14        |                0.681 |                0.805 | TP=49, TN=223, FP=54, FN=23  |
| method4   | logistic | smote            | 0.732±0.090 | 0.597±0.133  | 0.521±0.141         | 0.707±0.123      | 0.823±0.063   | 0.835±0.049 | 0.619±0.111 | 0.442±0.116      |                 72 |              14.4 |             0.49 | 15,15,14,14,14        |                0.708 |                0.823 | TP=51, TN=228, FP=49, FN=21  |
| method4   | rf       | adasyn           | 0.704±0.047 | 0.533±0.094  | 0.544±0.073         | 0.573±0.182      | 0.867±0.054   | 0.865±0.046 | 0.659±0.096 | 0.348±0.091      |                 72 |              14.4 |             0.49 | 15,15,14,14,14        |                0.569 |                0.866 | TP=41, TN=240, FP=37, FN=31  |
| method4   | rf       | borderline_smote | 0.718±0.054 | 0.552±0.105  | 0.567±0.068         | 0.588±0.195      | 0.877±0.049   | 0.865±0.046 | 0.656±0.114 | 0.363±0.168      |                 72 |              14.4 |             0.49 | 15,15,14,14,14        |                0.583 |                0.877 | TP=42, TN=243, FP=34, FN=30  |
| method4   | rf       | smote            | 0.726±0.044 | 0.565±0.075  | 0.591±0.103         | 0.571±0.135      | 0.888±0.052   | 0.863±0.043 | 0.675±0.103 | 0.390±0.141      |                 72 |              14.4 |             0.49 | 15,15,14,14,14        |                0.569 |                0.888 | TP=41, TN=246, FP=31, FN=31  |
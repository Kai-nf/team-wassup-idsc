# Implementation Plan: Phase 6 (Interpretability & Explainable AI)

## Constraints & Adaptations
The user requested Phase 6 execution which originally included:
1. SHAP Analysis for XGBoost/RF
2. Grad-CAM for 1D-CNN
3. Feature Importance Ranking

**CRITICAL ADAPTATION**: Based on the preceding work and explicit user rules, the `felicia` directory and `models.py` strictly **only support Logistic Regression and Random Forest**. The XGBoost and 1D-CNN models were explicitly stripped from the pipeline because they belong to other team members. 

Therefore, Phase 6 will be adapted as follows strictly for **Random Forest**:
1. **SHAP Analysis (Random Forest)**: Generate Beeswarm and Waterfall plots using the Random Forest model trained on Method 4 features.
2. **Feature Importance (Random Forest)**: Extract and plot the top 10 features from the trained Random Forest's `feature_importances_` attribute.
3. ~Grad-CAM (1D-CNN)~: Skipped, as CNNs are not supported by this pipeline.

---

## 6.1 SHAP Analysis & Feature Importance (Random Forest)

### Approach
Because we need to train on the *full dataset* (all 363 samples) for visualization purposes (no CV splitting), we will create a dedicated script `felicia/models/run_interpretability.py`. This script will:
1. Load the Method 4 dataset (`Steve/dataset_v4_features_drop14.csv`).
2. Train a tuned Random Forest classifier on the entire dataset.
3. Use the `shap` library (`TreeExplainer`) to compute SHAP values.
4. Generate the following publication-quality plots:
   - **Feature Importance Bar Chart**: Top 10 features based on RF `feature_importances_`.
   - **SHAP Summary Plot (Beeswarm)**: Showing the impact of the top features on the model output.
   - **SHAP Waterfall Plots**: For 2-3 specific patient predictions (e.g., one True Positive, one True Negative).
5. Save all plots to a new directory `felicia/results/interpretability/`.

### Dependencies
We need to ensure the `shap` library and `matplotlib` are installed in the `.venv`.
```bash
.venv/bin/pip install shap matplotlib
```

### Proposed Code `felicia/models/run_interpretability.py`
The script will mirror `models.py`'s data loading but skip nested CV.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import shap

# 1. Load Data
# We will load the Method 4 dataset directly.
df = pd.read_csv("../../Steve/dataset_v4_features_drop14.csv")
metadata = pd.read_csv("../../metadata.csv")

# Filter for drop14 patients
df = df[df["patient_id"].isin(metadata["patient_id"])]
X = df.drop(columns=["patient_id"])
feature_names = X.columns.tolist()

# Get binarized labels
def _binarize_labels(series):
    return series.astype(str).str.lower().map({"brugada": 1, "non-brugada": 0}).values
label_map = {int(p): l for p, l in zip(metadata["patient_id"], _binarize_labels(metadata["brugada"]))}
y = np.array([label_map[int(p)] for p in df["patient_id"]])

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Train RF on full dataset
# Using parameters determined from Phase 5 GridSearchCV
rf = RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_leaf=3, class_weight="balanced", random_state=42)
rf.fit(X_scaled, y)

# 3. Feature Importance Bar Chart
importances = rf.feature_importances_
indices = np.argsort(importances)[-10:] # Top 10
plt.figure(figsize=(10, 6))
plt.title("Top 10 Feature Importances (Random Forest)")
plt.barh(range(10), importances[indices], align="center")
plt.yticks(range(10), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.tight_layout()
plt.savefig("../../felicia/results/interpretability/rf_feature_importance.png", dpi=300)
plt.close()

# 4. SHAP summary
explainer = shap.TreeExplainer(rf)
# Random Forest predict_proba has shape (n_samples, 2), SHAP returns a list of arrays for trees.
# We focus on the positive class (index 1).
shap_values = explainer.shap_values(X_scaled)
if isinstance(shap_values, list): # For older shap versions or specific models
    shap_values = shap_values[1]

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_scaled, feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig("../../felicia/results/interpretability/rf_shap_beeswarm.png", dpi=300)
plt.close()

# 5. SHAP Waterfall
# Select a true positive and true negative patient
tp_idx = np.where((y == 1) & (rf.predict(X_scaled) == 1))[0][0]
tn_idx = np.where((y == 0) & (rf.predict(X_scaled) == 0))[0][0]

for name, idx in [("TruePositive", tp_idx), ("TrueNegative", tn_idx)]:
    plt.figure(figsize=(10, 6))
    exp = shap.Explanation(values=shap_values[idx], base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value, data=X_scaled[idx], feature_names=feature_names)
    shap.waterfall_plot(exp, show=False)
    plt.tight_layout()
    plt.savefig(f"../../felicia/results/interpretability/rf_shap_waterfall_{name}.png", dpi=300)
    plt.close()
```

### Verification
1. Run `pip install shap`.
2. Execute `run_interpretability.py`.
3. Verify the creation of `felicia/results/interpretability/` containing the PNG files.

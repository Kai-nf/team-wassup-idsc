# Team Wassup IDSC: Brugada Syndrome 12-Lead ECG Classification

## 🚀 Project Overview

This repository contains the machine learning pipeline and exploratory analysis for detecting Brugada Syndrome from standard 12-lead ECG signals. Because Brugada Syndrome often presents with hidden or transient ECG abnormalities, it is a leading cause of sudden cardiac death in otherwise healthy young adults. We implement a rigorous, end-to-end framework encompassing clinical signal processing, advanced feature engineering, and robust deep learning models **to transform a standard, low-cost ECG into a life-saving, early-warning screening tool.**

**Tech Stack**: PyTorch (1D-CNN), XGBoost, Random Forest, Logistic Regression.  
**Key Imbalance Mitigation**: SMOTE and ADASYN.

### Dataset Specifications

- Raw ECG source format: WFDB (`.hea` + `.dat`)
- Sampling frequency: 100 Hz
- Recording length: 12 seconds per recording
- Label source: [metadata.csv](metadata.csv)

## Current Standardized Setup

Phase 2 Method 1 and Method 2 are standardized using two experiment arms:

1. `drop14`: excludes the 14 Phase 1 signal-quality flagged recordings
2. `keepall`: keeps all recordings

For each arm:

1. Method 1 and Method 2 use the same cohort
2. Method 1 and Method 2 use the same 5-fold test assignments
3. Fold test IDs are matched exactly across Method 1 and Method 2

## Methods Implemented

### Method 1: Raw Baseline

Pipeline in [data_preprocessing/standard_clinical_preprocessing.py](data_preprocessing/standard_clinical_preprocessing.py):

1. Load all signals from [data_loader.py](data_loader.py)
2. Apply arm policy (`drop14` or `keepall`)
3. 5-fold stratified cross-validation split
4. Save fold-wise raw tensors and audit artifacts

### Method 2: Standard Clinical Preprocessing

Pipeline in [data_preprocessing/standard_clinical_preprocessing.py](data_preprocessing/standard_clinical_preprocessing.py):

1. Reuse Method 1 cohort and fold assignments per arm
2. Apply bandpass filter (0.5-40 Hz)
3. Apply notch filter (50 Hz)
4. Apply per-lead `StandardScaler` fit on train fold only, transform test fold
5. Save fold-wise filtered tensors

### Method 3: Deep Learning (1D-CNN)
1. Applies Continuous Wavelet Transform (CWT) to extract time-frequency features.
2. Trains a custom 1D-CNN evaluated at the patient level.
3. Pipeline located in `Steve/models/train_1dcnn_beat_level.py`.

### Method 4: Classical ML with Synthetic Balancing
1. Extracts clinical tabular features (intervals, amplitudes).
2. Applies SMOTE and ADASYN to aggressively balance the 79/21 class disparity.
3. Evaluates XGBoost, Random Forest, and Logistic Regression.

### The Ultimate Ensemble
Combines the deep learning feature extraction of Method 3 with the clinical rule-boundaries of Method 4 using a weighted soft-voting mechanism (CNN 0.65 + LR 0.35) tuned to a 0.35 threshold.

## 📊 Key Results

We rigorously evaluated multiple standalone architectures and ensembles using 5-fold stratified cross-validation (guaranteeing **zero data leakage** at the patient level). 

Our **Ultimate Model** is a **Weighted Soft-Voting Ensemble (CNN 0.65 + LR 0.35)** operating at a decision threshold of **0.35**. 

* **ROC-AUC**: **0.920**
* **Macro F1**: **0.818**
* **Clinical Viability**: Met all clinical sensitivity/specificity targets for life-saving screening tools.

## Generated Outputs

Running [main.py](main.py) generates these files in [data_preprocessing](data_preprocessing):

### Method 1 outputs

1. [data_preprocessing/dataset_v1_raw_drop14.npy](data_preprocessing/dataset_v1_raw_drop14.npy)
2. [data_preprocessing/dataset_v1_raw_drop14_manifest.csv](data_preprocessing/dataset_v1_raw_drop14_manifest.csv)
3. [data_preprocessing/dataset_v1_raw_drop14_dropped_ids.csv](data_preprocessing/dataset_v1_raw_drop14_dropped_ids.csv)
4. [data_preprocessing/dataset_v1_raw_drop14_summary.json](data_preprocessing/dataset_v1_raw_drop14_summary.json)
5. [data_preprocessing/dataset_v1_raw_keepall.npy](data_preprocessing/dataset_v1_raw_keepall.npy)
6. [data_preprocessing/dataset_v1_raw_keepall_manifest.csv](data_preprocessing/dataset_v1_raw_keepall_manifest.csv)
7. [data_preprocessing/dataset_v1_raw_keepall_dropped_ids.csv](data_preprocessing/dataset_v1_raw_keepall_dropped_ids.csv)
8. [data_preprocessing/dataset_v1_raw_keepall_summary.json](data_preprocessing/dataset_v1_raw_keepall_summary.json)

### Method 2 outputs

1. [data_preprocessing/dataset_v2_filtered_drop14.npy](data_preprocessing/dataset_v2_filtered_drop14.npy)
2. [data_preprocessing/dataset_v2_filtered_keepall.npy](data_preprocessing/dataset_v2_filtered_keepall.npy)

### Method 3 outputs

1. Saved PyTorch model checkpoints (e.g., `.pt` files) generated in the `Steve/models/` directory.
2. Evaluation metrics, ROC-AUC curves, and 1D Grad-CAM visualization artifacts.

### Method 4 outputs

1. [Preprocessed_Dataset/dataset_v4_features_drop14.csv](Preprocessed_Dataset/dataset_v4_features_drop14.csv)
2. [Preprocessed_Dataset/dataset_v4_features_keepall.csv](Preprocessed_Dataset/dataset_v4_features_keepall.csv)
3. [fold_composition_v4_drop14.json](fold_composition_v4_drop14.json)
4. [fold_composition_v4_keepall.json](fold_composition_v4_keepall.json)
5. [Steve/method4_extraction_report_drop14.json](Steve/method4_extraction_report_drop14.json)
6. [Steve/method4_extraction_report_keepall.json](Steve/method4_extraction_report_keepall.json)

## ⚙️ How To Run

1. **Install Dependencies**  
   Ensure you have Python 3.10+ installed.
   ```bash
   pip install -r requirements.txt
   ```

2. **Activate virtual environment & Run Pipeline**
   ```bash
   source .venv/bin/activate
   python main.py
   ```

## 🧠 Interpretability

To bridge the gap between black-box AI and clinical trust, we emphasize Explainable AI (XAI):
* **1D Grad-CAM (PyTorch)**: Calculates and projects activation gradients back onto the original temporal ECG signal, directly showing cardiologists the exact waveform regions (e.g., coved ST-segment elevations) that triggered a Brugada prediction.
* **SHAP (SHapley Additive exPlanations)**: Used for evaluating feature importance in our tabular models (XGBoost, Random Forest).

## EDA and Data Quality Artifacts

1. EDA notebook: [eda_report.ipynb](eda_report.ipynb)
2. Data integrity summary: [Data_Integrity_Report.md](Data_Integrity_Report.md)
3. Phase 1 signal-quality flags: [data_preprocessing/flagged_recordings_phase1.csv](data_preprocessing/flagged_recordings_phase1.csv)

## 📚 Mandatory Citations

1. García-Iglesias, D., Calvo, D., & de Cos, F. J. (2024). 12-lead ECGs of Brugada syndrome patients and controls (version 1.0.0). PhysioNet. https://doi.org/10.13026/0q9p-1474.
2. Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220.

## Notes

1. Large `.npy` artifacts are generated locally and are ignored by Git.
2. Re-run [main.py](main.py) any time to regenerate preprocessing outputs.
3. For fair ablation, Methods 3 and 4 should reuse the same arm policy and fold definitions.

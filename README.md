# Team Wassup IDSC: Brugada Syndrome 12-Lead ECG Classification

## 🚀 Project Overview

This repository implements a **complete machine learning pipeline** for detecting **Brugada Syndrome** from standard 12-lead ECG signals.

Brugada Syndrome is a major cause of sudden cardiac death in otherwise healthy individuals, often presenting with **transient or hidden ECG abnormalities**. This project aims to transform a **low-cost ECG into an early-warning screening tool** using both deep learning and clinical feature-based methods.

We fuse two complementary paradigms:

* 🧠 **1D-CNN (Deep Learning)** — learns beat-level morphology from wavelet-denoised ECG segments
* 📊 **Tabular ML Models** — SVM, XGBoost, Logistic Regression using handcrafted clinical features

A **strict inter-patient cross-validation framework** enforced via a centralised manifest ensures **zero data leakage**.

---

## 🧰 Tech Stack

* **Deep Learning**: PyTorch (1D-CNN)
* **Classical ML**: XGBoost, Random Forest, Logistic Regression, SVM
* **Signal Processing**: SciPy, WFDB, NeuroKit2
* **Imbalance Handling**: SMOTE, ADASYN
* **Visualization**: Matplotlib, Seaborn

---

## 📊 Dataset Specifications

* Format: WFDB (`.hea` + `.dat`)
* Sampling frequency: 100 Hz
* Duration: 12 seconds per recording
* Labels: `metadata.csv`
* Total patients (post-cleaning): **349**
* Brugada prevalence: **~21%**

---

## ⚙️ How To Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 2. Prepare Dataset

Place files into:

```
Environment_setup/
├── metadata.csv
└── files/
```

---

### 3. Run Preprocessing

```bash
python data_preprocessing/method1_raw_preprocessing.py
python data_preprocessing/method2_standard_clinical_preprocessing.py
python data_preprocessing/method3_wavelet_preprocessing.py
python data_preprocessing/method4_feature_engineering.py \
       Environment_setup/metadata.csv Environment_setup/files preprocessed_dataset
```

---

### 4. Train Models

```bash
python model/svm_trainer.py --resampler smote
python model/xgboost_trainer.py --resampler adasyn
python model/logistic_trainer.py --resampler smote
python model/1dcnn_beat_level.py --mode v3.1
```

---

### 5. Ensemble

```bash
python versatile_ensemble.py
```

Set weights:

```python
W_MODEL_A = 0.35
W_MODEL_B = 0.65
```

---

### 6. Aggregate Results

```bash
python aggregate_results.py
```

---

## 🧪 Standardized Experimental Setup

Two dataset arms:

| Arm       | Description                 |
| --------- | --------------------------- |
| `drop14`  | Removes 14 noisy recordings |
| `keepall` | Uses full dataset           |

All methods:

* Share identical patient splits
* Use same 5-fold CV
* Use central JSON manifest → **no leakage**

---

## 🧠 Methodology

### Method 1: Raw Baseline

* Loads raw ECG signals
* Applies cohort filtering (`drop14` / `keepall`)
* Saves fold-wise tensors

---

### Method 2: Clinical Preprocessing

Applies:

* **Bandpass filter (0.5–40 Hz)**
* **50 Hz notch filter**
* Per-lead normalization (train-only fit)

```python
from scipy.signal import butter, filtfilt, iirnotch
```

---

### Method 3: Deep Learning (1D-CNN)

Pipeline:

1. Wavelet denoising (db4, 4 levels)
2. R-peak detection (WFDB XQRS)
3. Beat segmentation (101 samples)

Input shape:

```
(N_beats, 101, 12)
```

CNN outputs are aggregated to patient-level via:

👉 **Max-rollup (clinically justified)**

---

### Method 4: Feature Engineering

Extracts **19 clinical features**:

* Intervals (QRS, PR, RR)
* Amplitudes
* Brugada-specific ST features

Uses:

```python
neurokit2
```

Missing values → median imputation (train-fold only)

---

## ⚖️ Handling Class Imbalance

| Model    | Strategy                  |
| -------- | ------------------------- |
| CNN      | `pos_weight ≈ 3.76`       |
| SVM      | `class_weight='balanced'` |
| XGBoost  | SMOTE / ADASYN            |
| Logistic | SMOTE                     |

---

## 🔄 Ensemble Architecture

Weighted soft voting:

```python
prob = w_a * tabular + w_b * cnn
```

CNN probabilities calibrated via:

```python
IsotonicRegression
```

---

## 📈 Key Results

### 🎯 Ultimate Model: CNN (0.65) + LR (0.35)

| Metric      | Value     |
| ----------- | --------- |
| ROC-AUC     | **0.920** |
| Macro F1    | **0.818** |
| Recall      | ≥ 0.80    |
| Specificity | ≥ 0.70    |

✅ Meets clinical screening thresholds

---

## 🧬 ECG Augmentation Strategy

Applied only during training:

| Type       | Examples            |
| ---------- | ------------------- |
| Amplitude  | Noise, gain scaling |
| Temporal   | Drift, shift, warp  |
| Structural | Channel dropout     |

⚠️ Constraint:

* **V1–V3 leads protected** (critical for Brugada detection)

---

## 📁 Generated Outputs

### Method 1

* `dataset_v1_raw_*.npy`
* manifest + summary files

### Method 2

* `dataset_v2_filtered_*.npy`

### Method 3

* `.pt` model checkpoints
* predictions JSON

### Method 4

* `dataset_v4_features_*.csv`
* fold JSONs
* extraction reports

### Final Outputs

* `results/master_ablation_table.csv`
* ensemble predictions

---

## 🧠 Interpretability & Clinical Safety

To bridge the gap between black-box AI and clinical trust, we emphasize Explainable AI (XAI) to ensure all predictions align with cardiological gold standards:

* **1D Grad-CAM (PyTorch):** Projects activation gradients back onto the temporal ECG signal. Our visual audits prove the CNN correctly anchors its attention on the QRS complex and ST-segment to identify the pathological Brugada "coved" pattern.
* **SHAP (SHapley Additive exPlanations):** Evaluates feature importance in our tabular models, verifying that clinical markers like patient age and QRS duration are weighted appropriately.
* **Failure Mode Analysis:** We explicitly map False Positives and False Negatives to understand our clinical boundaries, ensuring safe deployment as a screening aid rather than an autonomous diagnostic tool.

---

## 🔍 Data Quality & EDA

* `eda_report.ipynb`
* `Data_Integrity_Report.md`
* `flagged_recordings_phase1.csv`

---

## 🌟 Clinical Impact & "Hope"

Brugada Syndrome often hides in plain sight, making sudden cardiac death the first and only symptom for many young patients. By engineering a pipeline that is both highly sensitive and explicitly interpretable, we transform a standard, low-cost 12-lead ECG into a pervasive early-warning system. This model provides a digital safety net for primary care clinics, shifting diagnostic power away from expensive, reactive specialist centers to proactive, life-saving screening.

## 📚 Mandatory Citations

1. García-Iglesias, D., Calvo, D., & de Cos, F. J. (2024).  
   *12-lead ECGs of Brugada syndrome patients and controls (version 1.0.0).*  
   PhysioNet. https://doi.org/10.13026/0q9p-1474

2. Goldberger, A. L., Amaral, L. A. N., Glass, L., Hausdorff, J. M., Ivanov, P. C., Mark, R. G., ... & Stanley, H. E. (2000).  
   *PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals.*  
   Circulation, 101(23), e215–e220.

---

## ⚠️ Notes

* Large `.npy` files are ignored by Git
* Re-run preprocessing anytime
* All models must reuse the same folds for fair comparison

---

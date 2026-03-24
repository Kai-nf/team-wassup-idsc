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

Random Forest / Logistic Regression
```bash
python felicia/models/models.py --method {method1,method2,method3,method3_1,method3_2,method4} --model {logistic,rf} --resampler {none,smote,borderline_smote,adasyn}
```

SVM
```bash
python model/svm_trainer.py --version {v1,v2,v3,v4}
```

XGBoost
```bash
python model/models_xgboost.py --resampler {adasyn,smote,borderline_smote,none}
```

1D-CNN (beat-level)
```bash
python model/1d_cnn_beat_level.py --mode {v3,v3.1}
```

1D-CNN (patient-level)
```bash
python model/1d_cnn_patient_level.py 
```

### 5. Ensemble

```bash
python versatile_ensemble.py
```

* Set weights:
Adjust the proportional weight given to each model's prediction. Empirical testing shows that heavily favoring the CNN's spatial pattern recognition, while using the Logistic Regression as a morphological stabilizer, yields the highest ROC-AUC.
```python
W_MODEL_A = 0.35 # Weight for Tabular Model 
W_MODEL_B = 0.65 # Weight for Deep Learning Model
```

* Set thresholds:
In standard machine learning, the decision boundary is 0.5. However, Brugada syndrome is a life-threatening arrhythmia. To minimize False Negatives, we dynamically lower the decision threshold to prioritize Recall (Sensitivity), ensuring we flag high-risk patients even if the model is slightly uncertain.
```python
THRESHOLD = 0.35 (Any combined probability >= 0.35 is classified as Brugada)
```
---

### 6. Aggregate Results

```bash
python results/aggregate_results.py
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

Outputs Generated:
* dataset_v1_raw_drop14.npy: The core dataset containing the pre-split folds.
* dataset_v1_raw_drop14_manifest.csv: A complete audit trail of all patients, their inclusion status, and assigned test folds.
* dataset_v1_raw_drop14_dropped_ids.csv: A specific log of excluded patients for downstream tracking.
* dataset_v1_raw_drop14_summary.json: High-level metrics on retained/excluded records and class balances.

* Tensor Shape: (N_patients, 1000, 12) — representing 10 seconds of raw data at 100 Hz across 12 standard leads.

---

### Method 2: Clinical Preprocessing

Applies:

* **Bandpass filter (0.5–40 Hz)**
* **50 Hz notch filter**
* Per-lead normalization (train-only fit)
```python
from scipy.signal import butter, filtfilt, iirnotch
```

Outputs Generated:
* dataset_v2_filtered_drop14.npy: The core dataset containing the filtered, scaled, and pre-split folds.

* Tensor Shape: (N_patients, 1000, 12) — representing 10 seconds of filtered data at 100 Hz across 12 standard leads.

---

### Method 3: Deep Learning (1D-CNN)

Pipeline:

1. Wavelet denoising (db4, 4 levels) - Instead of standard clinical bandpass filters (Method 2), this uses a Daubechies 4 (db4) wavelet transform. It decomposes the signal into 4 levels, zeros out the lowest frequency (cA4) to completely flatten baseline wander, and uses statistical soft-thresholding to smooth out high-frequency noise.

2. R-peak detection (WFDB XQRS) - It runs the XQRS detection algorithm on every single lead. To avoid false positives, it requires a "consensus" where at least 3 separate leads agree on the exact location of an R-peak within a 0.2-second window (with safe fallbacks to 2 or 1 lead if necessary).

3. Beat segmentation (101 samples) - Once an R-peak is confirmed, the script acts like a cookie-cutter. It slices out exactly 0.5 seconds before and 0.5 seconds after the peak, resulting in a perfectly centered 1-second heartbeat.

Outputs Generated:
* dataset_v3.1_wavelet.npy: A pure multidimensional NumPy array containing all extracted heartbeats.

Tensor Shape:
* (N_total_beats, 101, 12) - representing 1 second of centered heartbeat data (50 samples before + R-peak + 50 samples after) across 12 leads.

Input shape:

```
(N_beats, 101, 12)
```

CNN outputs are aggregated to patient-level via:

👉 **Max-rollup (clinically justified)**

---

### Method 4: Feature Engineering

Extracts **19 clinical features**:

* Applies the exact same clinical Bandpass (0.5-40 Hz) and Notch (50 Hz) filters used in Method 2 to establish a clean baseline.

* Utilizes the neurokit2 library to perform automated ECG delineation . This finds the exact mathematical onset, peak, and offset of every P, Q, R, S, and T wave.

* Calculates 19 highly specific metrics, focusing heavily on leads V1, V2, and V3 (the precordial leads where Brugada morphology actually manifests). Features include ST-segment elevation, QRS duration, and T-wave symmetry.

* Uses "Median Imputation" as a safety net. If a signal is slightly messy and the algorithm fails to calculate one specific feature (resulting in a NaN), it fills that blank with the median value of the rest of the cohort so you don't lose the entire patient record.

Outputs Generated:
* dataset_v4_features_{arm_name}.csv: Contains the 19 numerical features and the final label for each patient.

Data Shape:
* (N_patients, 19) — Dimensionality is drastically reduced from the raw waveforms to focus strictly on human-readable clinical biomarkers.

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

### 👍 Recommended Operating Point

| Configuration | Threshold | Recall | Specificity | F1 | ROC-AUC | Context |
| ----------- | --------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| CNN 0.65 + LR 0.35 | 0.35 | 0.806 | 0.888 | 0.818 | 0.920 | Diagnostic support |
| CNN 0.65 + RF 0.35 | 0.25 | 0.875 | 0.805 | 0.771 | 0.920 | Population screening |

✅ Both configurations meet the clinical targets of recall ≥ 0.80 and specificity ≥ 0.70 simultaneously — a result no single model achieved in isolation.

### 🎯 Ultimate Model: CNN (0.65) + LR (0.35)

| Metric      | Value     |
| ----------- | --------- |
| ROC-AUC     | **0.920** |
| Macro F1    | **0.818** |
| Recall      | ≥ 0.80 (0.807)    |
| Specificity | ≥ 0.70 (0.884)    |

✅ Meets clinical screening thresholds

---

## 🧬 ECG Augmentation Strategy

The augmentation strategies are divided into three tiers:

| Type       | Examples            |
| ---------- | ------------------- |
| Amplitude  | Noise, gain scaling |
| Temporal   | Drift, shift, warp  |
| Structural | Channel dropout     |

* Amplitude tier (noise, gain) — fire independently. Scale amplitude only; never alter waveform shape. Safe to stack.
* Temporal tier (drift, shift, warp) — each individually safe, but stacking two or more distorts the ST segment non-linearly. At independent p=0.5 for each: 56.3% of beats previously received 3+ simultaneous temporal distortions (the Avalanche Effect). Fixed in v2 with a shared budget: exactly one of {drift, shift, warp} fires per beat.
* Structural tier (channel dropout) — independent of morphology, safe with any combination.

⚠️ Constraint:

* **V1–V3 leads (leads indices 6,7,8) protected** (critical for Brugada detection)

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

## 📚 Citations

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

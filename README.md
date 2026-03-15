# Team Wassup IDSC

## Project Overview

This repository contains the preprocessing and exploratory analysis pipeline for 12-lead ECG data used in Brugada syndrome classification.

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

## How To Run

1. Activate virtual environment
2. Run pipeline

```bash
source .venv/bin/activate
python main.py
```

## EDA and Data Quality Artifacts

1. EDA notebook: [eda_report.ipynb](eda_report.ipynb)
2. Data integrity summary: [Data_Integrity_Report.md](Data_Integrity_Report.md)
3. Phase 1 signal-quality flags: [data_preprocessing/flagged_recordings_phase1.csv](data_preprocessing/flagged_recordings_phase1.csv)

## Notes

1. Large `.npy` artifacts are generated locally and are ignored by Git.
2. Re-run [main.py](main.py) any time to regenerate preprocessing outputs.
3. For fair ablation, Methods 3 and 4 should reuse the same arm policy and fold definitions.

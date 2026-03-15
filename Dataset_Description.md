# Brugada Syndrome ECG Dataset

## Overview

This dataset contains electrocardiographic (ECG) recordings from 363 individuals with suspected Brugada Syndrome. Brugada syndrome is a rare but potentially life-threatening cardiac arrhythmia disorder, marked by distinctive ECG abnormalities and an elevated risk of sudden cardiac death.

## Background

Brugada syndrome is characterized by a coved-type ST-segment elevation in the right precordial leads (V1–V3), frequently accompanied by a right bundle branch block pattern. Diagnosis is primarily clinical and is based on the identification of this ECG pattern — either occurring spontaneously or induced by sodium channel blockers — along with clinical criteria such as a history of syncope, documented ventricular arrhythmias, or a family history of sudden cardiac death.

## Data Acquisition

- **Sampling Frequency**: 100 Hz
- **Recording Duration**: 12 seconds per subject
- **Number of Leads**: 12 standard ECG leads
- **Total Subjects**: 363 individuals

## Folder Structure

```
├── metadata.csv                 # Clinical info about subjects
├── metadata_dictionary.csv      # Dictionary explaining metadata variables
├── RECORDS                      # List of all patient IDs
├── files/                       # ECG data files organized by patient ID
│   ├── 188981/
│   │   ├── 188981.dat         # ECG signal data
│   │   └── 188981.hea         # Header file with recording metadata
│   ├── 251972/
│   │   ├── 251972.dat
│   │   └── 251972.hea
│   └── [...]
```

## Data Characteristics

### Subject Distribution

Based on the `brugada` field in metadata:
- **0**: Healthy individuals
- **1**: Confirmed Brugada Syndrome diagnosis
- **2**: Other/atypical cases

### Clinical Variables

- **basal_pattern**: Indicates pathological baseline ECG patterns (independent of Brugada diagnosis)
- **sudden_death**: Critical outcome variable for risk assessment
- **brugada**: Primary diagnostic label

## Derived Preprocessing Datasets

The project currently maintains standardized Phase 2 outputs for Method 1 and Method 2 in two experiment arms.

### Arm definitions

1. `drop14`: Exclude 14 Phase 1 signal-quality flagged recordings
2. `keepall`: Keep all recordings

### Method 1 outputs (raw baseline)

1. [data_preprocessing/dataset_v1_raw_drop14.npy](data_preprocessing/dataset_v1_raw_drop14.npy)
2. [data_preprocessing/dataset_v1_raw_keepall.npy](data_preprocessing/dataset_v1_raw_keepall.npy)

Method 1 audit artifacts:

1. [data_preprocessing/dataset_v1_raw_drop14_manifest.csv](data_preprocessing/dataset_v1_raw_drop14_manifest.csv)
2. [data_preprocessing/dataset_v1_raw_drop14_dropped_ids.csv](data_preprocessing/dataset_v1_raw_drop14_dropped_ids.csv)
3. [data_preprocessing/dataset_v1_raw_drop14_summary.json](data_preprocessing/dataset_v1_raw_drop14_summary.json)
4. [data_preprocessing/dataset_v1_raw_keepall_manifest.csv](data_preprocessing/dataset_v1_raw_keepall_manifest.csv)
5. [data_preprocessing/dataset_v1_raw_keepall_dropped_ids.csv](data_preprocessing/dataset_v1_raw_keepall_dropped_ids.csv)
6. [data_preprocessing/dataset_v1_raw_keepall_summary.json](data_preprocessing/dataset_v1_raw_keepall_summary.json)

### Method 2 outputs (clinical filtering)

1. [data_preprocessing/dataset_v2_filtered_drop14.npy](data_preprocessing/dataset_v2_filtered_drop14.npy)
2. [data_preprocessing/dataset_v2_filtered_keepall.npy](data_preprocessing/dataset_v2_filtered_keepall.npy)

Method 2 processing steps:

1. Bandpass filter: 0.5-40 Hz
2. Notch filter: 50 Hz
3. Per-lead standardization using train-fold fit only

### Cross-validation protocol

1. 5-fold stratified cross-validation
2. Method 2 reuses Method 1 fold assignments within each arm
3. Method 1 and Method 2 therefore have matched test IDs fold-by-fold for fair comparison

## Loading the Dataset

### Reading Metadata

```python
import pandas as pd

# Load metadata
metadata = pd.read_csv('metadata.csv')

# Load data dictionary
data_dict = pd.read_csv('metadata_dictionary.csv')

# Display basic statistics
print(metadata.head())
print(f"Total subjects: {len(metadata)}")
print(f"Brugada patients: {(metadata['brugada'] > 0).sum()}")
print(f"Healthy subjects: {(metadata['brugada'] == 0).sum()}")
```

### Reading ECG Signal Data

The ECG data is stored in WFDB (WaveForm DataBase) format, which is commonly used for physiological signals. You can use the `wfdb` Python package to read these files:

```python
import wfdb
import matplotlib.pyplot as plt

# Read a single patient's ECG
patient_id = '188981'
record = wfdb.rdrecord(f'files/{patient_id}/{patient_id}')

# Access the signal data
signals = record.p_signal  # Shape: (1200, 12) for 12s at 100Hz, 12 leads
lead_names = record.sig_name  # Lead names (I, II, III, aVR, aVL, aVF, V1-V6)
sampling_freq = record.fs  # Sampling frequency (100 Hz)

# Plot a specific lead
plt.figure(figsize=(12, 4))
plt.plot(signals[:, 0])  # Plot first lead
plt.title(f'Patient {patient_id} - {lead_names[0]}')
plt.xlabel('Sample')
plt.ylabel('Amplitude (mV)')
plt.show()
```
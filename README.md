# Team Wassup IDSC roadmap

## Pre-development Data Handling

### 1. Load Metadata & WFDB Pipelining

In `data_loader.py`, we defined a function to load the raw dataset from `metadata.csv`. A special class `wfdb` is imported here to convert the waveform records into computer-understandable formats for future ML development.
- `.hea` files are read first for the computer to understand the "map" of the data.
- `.dat` files are then read to capture the actual voltage values.
- If either file is missing, `wfdb.rdrecord` will throw an error. (which doesn't happen in this case)

### 2. Data Preprocessing

#### 2.1 Raw Baseline (Conservative)

#### 2.2 Standard Clinical Preprocessing

- Bandpass filter: 0.5–40 Hz (removes baseline wander + high-frequency noise)
- Notch filter: 50 Hz (removes powerline interference)
- Feature scaling: StandardScaler applied per-lead (fit on train only, transform on test)
- Holdout split: 70% train / 30% test, stratified by class
- Save as `dataset_v2_filtered.npy`

### 3. Exploratory Data Analysis (EDA)

The EDA report comprises of 6 separate parts to understand the dataset better:

- Data integrity check: Check for missing values in training dataset or duplicate patient_id entries.
- Class distribution: Check for class imbalance to prevent the trained model cheat by giving a response of all negative to attain high prediction accuracy.
- Signal Quality Assessment (SQA): Provide a simple visuals of a single lead, V1 (most important)
- Lead-level inspection: Compare normal and Brugada subjects accross leads **V1, V2 and V3**
- Correlation analysis: `corr()` calculates how much the leads move together.
- Descriptive statistics: Provide a clear view of the dataset characteristics through mathematical statistics.

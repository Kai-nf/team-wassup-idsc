# Brugada Syndrome ECG Classification Project

## Overview

This project aims to classify ECG signals for Brugada Syndrome using various preprocessing methods and feature extraction techniques. The dataset consists of ECG recordings from 363 patients, and the classification is based on morphological features extracted from the ECG signals.

## Project Structure

The project is organized as follows:

```
brugada-ecg-classifier
├── data_preprocessing
│   ├── jk_Method_3
│   │   └── method3_advanced_wavelet_pipeline.py
│   ├── method4_feature_engineering.py
│   └── standard_clinical_preprocessing.py
├── data_loader.py
├── metadata.csv
├── requirements.txt
└── README.md
```

### File Descriptions

- **data_preprocessing/method4_feature_engineering.py**: Implements the Method 4 (Feature Engineering) preprocessing pipeline. It includes:
  - `build_method4_feature_dataset`: Orchestrates the feature extraction process.
  - `extract_features_single_patient`: Extracts 19 morphological features per patient.
  - `load_feature_dataset_for_fold`: Prepares the dataset for cross-validation, handling signal preprocessing, feature extraction, NaN handling, and outputs results in specified CSV and JSON formats.

- **data_preprocessing/jk_Method_3/method3_advanced_wavelet_pipeline.py**: Contains the implementation of Method 3, focusing on wavelet denoising and beat segmentation of ECG signals.

- **data_preprocessing/standard_clinical_preprocessing.py**: Implements Methods 1 and 2, which involve raw baseline and bandpass/notch filtering of ECG signals.

- **data_loader.py**: Responsible for loading raw WFDB data into a structured numpy array format.

- **metadata.csv**: Contains metadata for 363 patients, including patient IDs and clinical information.

- **requirements.txt**: Lists the dependencies required for the project, including the neurokit2 library for ECG analysis.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd brugada-ecg-classifier
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the feature extraction pipeline, execute the following command:
```
python data_preprocessing/method4_feature_engineering.py
```

This will generate the feature dataset and the corresponding fold compositions in the specified output directory.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
from data_loader import load_raw_dataset
from data_preprocessing.standard_clinical_preprocessing import (
    run_preprocessing_pipeline,
    run_raw_baseline_pipeline,
)

# 1. Configuration
CSV_PATH = 'metadata.csv'
DATA_DIR = 'files'  # The folder containing your patient_id subfolders

def main():
    print("--- Phase 1: Loading Data ---")
    # This calls the loader we wrote earlier
    all_signals, labels = load_raw_dataset(CSV_PATH, DATA_DIR)
    print(f"Loaded {all_signals.shape[0]} recordings.")

    print("\n--- Phase 2: Method 1 (Raw Baseline) ---")
    folds_v1 = run_raw_baseline_pipeline(
        all_signals,
        labels,
        metadata_csv_path=CSV_PATH,
        flagged_csv_path='data_preprocessing/flagged_recordings_phase1.csv',
    )
    print("'dataset_v1_raw.npy' has been created.")
    print(f"Raw baseline: {len(folds_v1)} folds | fold 0 -> train: {len(folds_v1[0]['X_train'])}, test: {len(folds_v1[0]['X_test'])}")

    print("\n--- Phase 2: Method 2 (Clinical Filtered) ---")
    folds_v2 = run_preprocessing_pipeline(
        all_signals,
        labels,
        metadata_csv_path=CSV_PATH,
        dropped_csv_path='data_preprocessing/dataset_v1_raw_dropped_ids.csv',
        manifest_csv_path='data_preprocessing/dataset_v1_raw_manifest.csv',
    )
    print("'dataset_v2_filtered.npy' has been created.")
    print(f"Clinical filtered: {len(folds_v2)} folds | fold 0 -> train: {len(folds_v2[0]['X_train'])}, test: {len(folds_v2[0]['X_test'])}")

if __name__ == "__main__":
    main()
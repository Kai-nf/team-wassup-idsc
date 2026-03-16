from data_loader import load_raw_dataset
from data_preprocessing.method2 import run_preprocessing_pipeline
from felicia.data_preprocessing_method1.method1 import run_raw_baseline_pipeline

# 1. Configuration
CSV_PATH = 'metadata.csv'
DATA_DIR = 'files'  # The folder containing your patient_id subfolders

def main():
    print("--- Phase 1: Loading Data ---")
    # This calls the loader we wrote earlier
    all_signals, labels = load_raw_dataset(CSV_PATH, DATA_DIR)
    print(f"Loaded {all_signals.shape[0]} recordings.")

    arms = [
        {'name': 'drop14', 'drop_flagged_in_phase1': True},
        {'name': 'keepall', 'drop_flagged_in_phase1': False},
    ]

    for arm in arms:
        arm_name = arm['name']
        print(f"\n--- Phase 2: Method 1 (Raw Baseline) [{arm_name}] ---")
        folds_v1 = run_raw_baseline_pipeline(
            all_signals,
            labels,
            metadata_csv_path=CSV_PATH,
            flagged_csv_path='flagged_recordings_phase1.csv',
            output_dir='felicia/data_preprocessing_method1',
            arm_name=arm_name,
            drop_flagged_in_phase1=arm['drop_flagged_in_phase1'],
        )
        print(f"'dataset_v1_raw_{arm_name}.npy' has been created.")
        print(
            f"Raw baseline [{arm_name}]: {len(folds_v1)} folds | "
            f"fold 0 -> train: {len(folds_v1[0]['X_train'])}, test: {len(folds_v1[0]['X_test'])}"
        )

        print(f"\n--- Phase 2: Method 2 (Clinical Filtered) [{arm_name}] ---")
        folds_v2 = run_preprocessing_pipeline(
            all_signals,
            labels,
            metadata_csv_path=CSV_PATH,
            dropped_csv_path=f'felicia/data_preprocessing_method1/dataset_v1_raw_{arm_name}_dropped_ids.csv',
            manifest_csv_path=f'felicia/data_preprocessing_method1/dataset_v1_raw_{arm_name}_manifest.csv',
            arm_name=arm_name,
        )
        print(f"'dataset_v2_filtered_{arm_name}.npy' has been created.")
        print(
            f"Clinical filtered [{arm_name}]: {len(folds_v2)} folds | "
            f"fold 0 -> train: {len(folds_v2[0]['X_train'])}, test: {len(folds_v2[0]['X_test'])}"
        )

if __name__ == "__main__":
    main()
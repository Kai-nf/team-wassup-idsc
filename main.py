from data_loader import load_raw_dataset
from data_preprocessing.standard_clinical_preprocessing import run_preprocessing_pipeline

# 1. Configuration
CSV_PATH = 'metadata.csv'
DATA_DIR = 'files'  # The folder containing your patient_id subfolders

def main():
    print("--- Phase 1: Loading Data ---")
    # This calls the loader we wrote earlier
    all_signals, labels = load_raw_dataset(CSV_PATH, DATA_DIR)
    print(f"Loaded {all_signals.shape[0]} recordings.")

    print("\n--- Phase 2: Preprocessing ---")
    # This calls the pipeline: Split -> Filter -> Scale -> Save
    X_train, X_test = run_preprocessing_pipeline(all_signals, labels)
    
    print("\n'dataset_v2_filtered.npy' has been created.")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

if __name__ == "__main__":
    main()
import json
from pathlib import Path

def main():
    # 1. Navigate to the root folder (one level up from Steve/)
    REPO_ROOT = Path(__file__).resolve().parents[1]
    
    # 2. Point exactly to where the manifest lives in the JiaKang folder
    manifest_path = REPO_ROOT  / "master_folds_drop14.json"
    
    if not manifest_path.exists():
        print(f"Error: Could not find manifest file at {manifest_path}")
        print("Please check if the folder name is exactly 'JiaKang'")
        return

    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    print("\n### Cross-Validation Data Leakage Verification\n")
    print(f"| {'Fold':<5} | {'Train Patients':<15} | {'Test Patients':<15} | {'Overlap':<10} |")
    print(f"| {'-'*5} | {'-'*15} | {'-'*15} | {'-'*10} |")

    total_overlap = 0

    # 3. Loop through folds and check for leakage
    for fold_id in range(5):
        fold_key = str(fold_id)
        
        # Extract patient IDs and convert to sets for fast comparison
        train_pids = set(manifest["folds"][fold_key]["train"]["patient_ids"])
        test_pids = set(manifest["folds"][fold_key]["test"]["patient_ids"])
        
        # Find the intersection (any IDs that exist in both sets)
        overlap = train_pids.intersection(test_pids)
        overlap_count = len(overlap)
        total_overlap += overlap_count
        
        # Print the row for this fold
        print(f"| {fold_id:<5} | {len(train_pids):<15} | {len(test_pids):<15} | {overlap_count:<10} |")

    # 4. Print the final verdict for the judges
    print("\n" + "="*50)
    if total_overlap == 0:
        print("✅ DATA LEAKAGE CHECK PASSED: 0 Overlapping Patients")
    else:
        print(f"❌ WARNING: Found {total_overlap} overlapping patients across folds!")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
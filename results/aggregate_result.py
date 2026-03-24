import json
import numpy as np
import pandas as pd
from pathlib import Path

def extract_metrics(data: dict) -> dict:
    """
    Intelligently extract metrics. Tries the 'summary' block first.
    If unavailable, dives into 'folds' to manually calculate the mean.
    """
    target_metrics = ["macro_f1", "recall", "precision", "specificity", "roc_auc"]
    extracted = {m: None for m in target_metrics}

    # 1. Try extracting from a well-formatted "summary" block first
    if "summary" in data and isinstance(data["summary"], dict):
        summary = data["summary"]
        # Check if it uses the "_mean" naming convention (like your ensembler)
        if "macro_f1_mean" in summary:
            for m in target_metrics:
                val = summary.get(f"{m}_mean")
                if val is not None:
                    extracted[m] = float(val)
            return extracted

    # 2. Fallback: Manually calculate the mean by iterating through the "folds"
    if "folds" in data and isinstance(data["folds"], dict):
        temp_tracker = {m: [] for m in target_metrics}
        
        # Loop through "0", "1", "2", "3", "4"
        for fold_key, fold_data in data["folds"].items():
            if not isinstance(fold_data, dict):
                continue
            for m in target_metrics:
                if m in fold_data and fold_data[m] is not None:
                    temp_tracker[m].append(float(fold_data[m]))
        
        # Calculate the mean for each metric across the folds
        for m in target_metrics:
            if temp_tracker[m]:
                extracted[m] = float(np.mean(temp_tracker[m]))
                
    return extracted

def main():
    # Define where your results live
    repo_root = Path(__file__).resolve().parent.parent
    results_dir = repo_root / "results"
    
    if not results_dir.exists():
        print(f"[Error] Could not find the results directory at: {results_dir}")
        return

    # Find all JSON files, including those inside subfolders
    json_files = list(results_dir.rglob("*.json"))
    print(f"Found {len(json_files)} JSON files. Analyzing...\n")

    records = []

    for file_path in json_files:
        # Skip known non-evaluation files
        if "fold_composition" in file_path.name or "summary" in file_path.name:
            continue
            
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"  [Warning] Skipping {file_path.name}: Invalid JSON.")
            continue

        # Skip files that don't have evaluation data
        if "folds" not in data and "summary" not in data:
            continue

        # Extract the metrics safely
        metrics = extract_metrics(data)
        
        # Add to our leaderboard record
        records.append({
            "Model Name": file_path.stem, # .stem removes the '.json' extension
            "Macro F1": metrics["macro_f1"],
            "Recall": metrics["recall"],
            "Precision": metrics["precision"],
            "Specificity": metrics["specificity"],
            "ROC-AUC": metrics["roc_auc"]
        })

    if not records:
        print("No valid evaluation metrics found in the JSON files.")
        return

    # Convert to a Pandas DataFrame for beautiful sorting and formatting
    df = pd.DataFrame(records)
    
    # Sort by Macro F1 descending (best model at the top)
    df = df.sort_values(by="Macro F1", ascending=False).reset_index(drop=True)
    
    # Round all numbers to 4 decimal places for readability
    df = df.round(4)

    # Print the leaderboard to the terminal
    print("=" * 90)
    print(f"{'🏆 BRUGADA SYNDROME MODEL LEADERBOARD 🏆':^90}")
    print("=" * 90)
    print(df.to_string(index=True))
    print("=" * 90)

    # Save to CSV for easy copy-pasting into your presentation/README
    output_csv = repo_root / "master_leaderboard.csv"
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Leaderboard saved to: {output_csv.name}")

if __name__ == "__main__":
    main()
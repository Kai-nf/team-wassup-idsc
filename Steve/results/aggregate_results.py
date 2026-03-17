import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score

def aggregate_results():
    # 1. Define Paths strictly
    REPO_ROOT = Path(__file__).resolve().parents[2]
    results_dir = REPO_ROOT / "results"
    
    # 2. Define the models and their files
    models = {
        "Method 3 (1D-CNN)": "method3_1dcnn_predictions.json",
        "Method 4 (XGBoost)": "method4_xgboost_predictions.json",
        "Ensemble (CNN + XGB)": "ensemble_predictions.json"
    }
    
    table_data = []
    
    # Helper function to format the mean and std deviation
    def format_metric(scores):
        if not scores: 
            return "N/A"
        return f"{np.mean(scores):.3f} ± {np.std(scores):.3f}"

    # 3. Extract data and calculate metrics
    for model_name, filename in models.items():
        file_path = results_dir / filename
        if not file_path.exists():
            print(f"Warning: Missing tracking file for '{model_name}': {filename}. Skipping.")
            continue
            
        with open(file_path, "r") as f:
            data = json.load(f)
            
        f1_list, rec_list, prec_list, auc_list = [], [], [], []
        
        for fold_id in range(5):
            fold_key = f"fold_{fold_id}"
            if fold_key not in data:
                continue
                
            y_true = np.array(data[fold_key]["y_true_patient"])
            y_pred = np.array(data[fold_key]["y_pred_patient"])
            y_prob = np.array(data[fold_key]["y_prob_patient"])
            
            f1_list.append(f1_score(y_true, y_pred, average='macro', zero_division=0))
            rec_list.append(recall_score(y_true, y_pred, zero_division=0))
            prec_list.append(precision_score(y_true, y_pred, zero_division=0))
            
            if len(np.unique(y_true)) > 1:
                auc_list.append(roc_auc_score(y_true, y_prob))
                
        table_data.append({
            "Model": model_name,
            "F1 (Macro)": format_metric(f1_list),
            "Recall": format_metric(rec_list),
            "Precision": format_metric(prec_list),
            "ROC AUC": format_metric(auc_list)
        })

    # 4. Print nicely formatted Markdown Table to console
    print("\n### Phase 7: Final Ablation Results\n")
    print(f"| {'Model':<20} | {'F1 (Macro)':<15} | {'Recall':<15} | {'Precision':<15} | {'ROC AUC':<15} |")
    print(f"| {'-'*20} | {'-'*15} | {'-'*15} | {'-'*15} | {'-'*15} |")
    
    for row in table_data:
        print(f"| {row['Model']:<20} | {row['F1 (Macro)']:<15} | {row['Recall']:<15} | {row['Precision']:<15} | {row['ROC AUC']:<15} |")
    print("\n")

    # 5. Convert to Pandas DataFrame and Export
    df = pd.DataFrame(table_data)
    
    # Ensure the results directory exists
    results_dir.mkdir(parents=True, exist_ok=True)

    # Export to CSV
    csv_path = results_dir / "master_ablation_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"Successfully saved populated table to: {csv_path}")

    # Export to LaTeX
    tex_path = results_dir / "master_ablation_table.tex"
    df.to_latex(tex_path, index=False)
    print(f"Successfully saved populated LaTeX table to: {tex_path}")

if __name__ == "__main__":
    aggregate_results()
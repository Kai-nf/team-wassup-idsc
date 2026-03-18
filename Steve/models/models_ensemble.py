import json
import numpy as np
import os
from pathlib import Path
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score

def run_ensemble():
    REPO_ROOT = Path(__file__).resolve().parents[2]
    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    cnn_path = results_dir / "method3_1dcnn_predictions.json"
    xgb_path = results_dir / "method4_xgboost_predictions.json"
    output_path = results_dir / "ensemble_predictions.json"
    
    with open(cnn_path, 'r') as f:
        cnn_preds = json.load(f)
    with open(xgb_path, 'r') as f:
        xgb_preds = json.load(f)
        
    all_fold_preds = {}
    
    for fold_id in range(5):
        fold_key = f"fold_{fold_id}"
        
        # Extract raw lists
        cnn_data = cnn_preds[fold_key]
        xgb_data = xgb_preds[fold_key]
        
        cnn_pids = cnn_data["patient_ids"]
        cnn_probs = cnn_data["y_prob_patient"]
        cnn_trues = cnn_data["y_true_patient"]
        
        xgb_pids = xgb_data["patient_ids"]
        xgb_probs = xgb_data["y_prob_patient"]

        # Map patient_id (forced to integer) -> probability
        cnn_dict = {int(pid): prob for pid, prob in zip(cnn_pids, cnn_probs)}
        xgb_dict = {int(pid): prob for pid, prob in zip(xgb_pids, xgb_probs)}
        true_dict = {int(pid): true for pid, true in zip(cnn_pids, cnn_trues)}

        # 1. Assert the SET of patients match (ignoring order entirely)
        assert set(cnn_dict.keys()) == set(xgb_dict.keys()), f"Patient mismatch in {fold_key}!"

        # 2. Sort the integer IDs to create a master aligned order
        aligned_pids = sorted(cnn_dict.keys())
        
        # 3. Rebuild the probability arrays using this strict, aligned order
        aligned_cnn_probs = np.array([cnn_dict[pid] for pid in aligned_pids])
        aligned_xgb_probs = np.array([xgb_dict[pid] for pid in aligned_pids])
        aligned_trues = np.array([true_dict[pid] for pid in aligned_pids])

        # 4. Ensemble Logic (Soft Voting)
        ensemble_prob = (aligned_cnn_probs + aligned_xgb_probs) / 2.0
        ensemble_pred = (ensemble_prob >= 0.5).astype(int)

        # 5. Evaluation using aligned_trues
        f1 = f1_score(aligned_trues, ensemble_pred, average='macro', zero_division=0)
        rec = recall_score(aligned_trues, ensemble_pred, zero_division=0)
        prec = precision_score(aligned_trues, ensemble_pred, zero_division=0)
        
        if len(np.unique(aligned_trues)) > 1:
            auc = roc_auc_score(aligned_trues, ensemble_prob)
        else:
            auc = float('nan')
            
        print(f"{fold_key} Ensemble -> F1(macro): {f1:.3f} | Recall: {rec:.3f} | Precision: {prec:.3f} | AUC: {auc:.3f}")

        # 6. Save back to the output dictionary
        all_fold_preds[fold_key] = {
            "patient_ids": aligned_pids,
            "y_true_patient": aligned_trues.tolist(),
            "y_prob_patient": ensemble_prob.tolist(),
            "y_pred_patient": ensemble_pred.tolist()
        }
        
    with open(output_path, 'w') as f:
        json.dump(all_fold_preds, f, indent=4)
        
    print(f"\nSaved ensemble predictions to: {output_path}")

if __name__ == "__main__":
    run_ensemble()
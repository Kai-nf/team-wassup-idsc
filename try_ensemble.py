"""
ensemble_voting.py
==================
The Grand Finale: Multi-Modal Soft Voting Ensemble
Combines JiaKang's 1D-CNN (Raw Waveforms) with Steve's XGBoost (Clinical Features).
"""

import json
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix

def main():
    print("=" * 60)
    print(" 🚀 INITIATING MULTI-MODAL SOFT VOTING ENSEMBLE")
    print("  Model A: 1D-CNN (Patient-Level Rollup on v3.1)")
    print("  Model B: XGBoost (ADASYN + Tabular Features v4)")
    print("=" * 60)

    REPO_ROOT = Path(__file__).resolve().parent
    
    # ---------------------------------------------------------
    # IMPORTANT: Update these filenames to match your exact output JSONs
    # ---------------------------------------------------------
    path_cnn = REPO_ROOT / "results" / "method3.1_1dcnn_beat_level.json" 
    path_xgb = REPO_ROOT / "results" / "method4_xgboost_predictions.json"
    
    if not path_cnn.exists() or not path_xgb.exists():
        print("[ERROR] Could not find both prediction JSON files.")
        print(f"Looking for: \n1. {path_cnn}\n2. {path_xgb}")
        return

    with open(path_cnn, 'r') as f:
        cnn_data = json.load(f)
    with open(path_xgb, 'r') as f:
        xgb_data = json.load(f)

    metrics = {"macro_f1": [], "recall": [], "precision": [], "roc_auc": []}

    for fold_id in range(5):
        fold_key = str(fold_id)
        
        # Accessing the XGB predictions
        xgb_fold = xgb_data[f"fold_{fold_id}"]
        xgb_pids = xgb_fold["patient_ids"]
        xgb_probs = xgb_fold["y_prob_patient"]
        xgb_trues = xgb_fold["y_true_patient"]
        
        # Accessing the CNN predictions (Adjust key access based on your CNN JSON schema)
        # Assuming the CNN script saved them under 'folds' -> '0' -> 'y_prob_patient' etc.
        cnn_fold = cnn_data["folds"][fold_key]
        cnn_pids = cnn_fold["patient_ids"]
        cnn_probs = cnn_fold["y_prob_patient"]

        # Create dictionaries to map patient_id -> probability for safe matching
        dict_xgb = {pid: prob for pid, prob in zip(xgb_pids, xgb_probs)}
        dict_cnn = {pid: prob for pid, prob in zip(cnn_pids, cnn_probs)}
        dict_true = {pid: true_val for pid, true_val in zip(xgb_pids, xgb_trues)}

        # Find common patients (should be identical, but safe to intersect)
        common_pids = set(dict_xgb.keys()).intersection(set(dict_cnn.keys()))
        
        y_true_ensemble = []
        y_prob_ensemble = []
        
        for pid in common_pids:
            # SOFT VOTING: Average the probabilities
            avg_prob = (dict_cnn[pid] + dict_xgb[pid]) / 2.0
            
            y_prob_ensemble.append(avg_prob)
            y_true_ensemble.append(dict_true[pid])

        # Convert to numpy arrays
        y_true_ensemble = np.array(y_true_ensemble)
        y_prob_ensemble = np.array(y_prob_ensemble)
        
        # Generate final prediction (Threshold = 0.5)
        y_pred_ensemble = (y_prob_ensemble >= 0.5).astype(int)

        # Calculate Fold Metrics
        f1_mac = f1_score(y_true_ensemble, y_pred_ensemble, average='macro')
        recall = recall_score(y_true_ensemble, y_pred_ensemble)
        precision = precision_score(y_true_ensemble, y_pred_ensemble, zero_division=0)
        roc_auc = roc_auc_score(y_true_ensemble, y_prob_ensemble)
        tn, fp, fn, tp = confusion_matrix(y_true_ensemble, y_pred_ensemble).ravel()

        print(f"\n--- Fold {fold_id} (N={len(common_pids)}) ---")
        print(f"F1 (Macro): {f1_mac:.4f} | Recall: {recall:.4f} | Precision: {precision:.4f} | ROC AUC: {roc_auc:.4f}")
        print(f"Confusion Matrix: TP:{tp} FN:{fn} TN:{tn} FP:{fp}")

        metrics["macro_f1"].append(f1_mac)
        metrics["recall"].append(recall)
        metrics["precision"].append(precision)
        metrics["roc_auc"].append(roc_auc)

    print("\n" + "=" * 50)
    print(" 🏆 FINAL ENSEMBLE CROSS-VALIDATION SUMMARY")
    print("=" * 50)
    for k in metrics.keys():
        vals = metrics[k]
        print(f"  {k:<10}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

if __name__ == "__main__":
    main()
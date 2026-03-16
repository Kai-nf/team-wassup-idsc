import json
import numpy as np
import os
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score

def run_ensemble():
    os.makedirs("results", exist_ok=True)
    
    # Load predictions (placeholders)
    # preds_model1 = load_predictions("rf_preds.npy")
    # preds_model2 = load_predictions("1d_cnn_preds.npy")
    
    ensemble_results = {}
    
    for fold_id in range(5):
        # Averages the predicted probabilities
        # prob_ensemble = (preds_model1[fold_id] + preds_model2[fold_id]) / 2.0
        # preds_binary = (prob_ensemble >= 0.5).astype(int)
        
        # Compute metrics
        ensemble_results[fold_id] = {
            "f1_macro": 0.88, # placeholder
            "roc_auc": 0.92
        }
        
    with open("master_folds_drop14.json") as f:
        manifest = json.load(f)

if __name__ == "__main__":
    run_ensemble()
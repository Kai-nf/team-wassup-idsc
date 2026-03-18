import json
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
from xgboost import XGBClassifier

def main():
    REPO_ROOT = Path(__file__).resolve().parents[2]
    
    data_path = REPO_ROOT / "Steve" / "dataset_v4_features_drop14.csv"
    folds_path = REPO_ROOT / "JiaKang"/"master_folds_drop14.json"
    
    output_dir = REPO_ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "method4_xgboost_predictions.json"

    # Load data
    df = pd.read_csv(data_path)
    
    # Load folds
    with open(folds_path, 'r') as f:
        master_folds = json.load(f)

    all_fold_preds = {}

    for fold_id in range(5):
        print(f"\n--- Fold {fold_id} ---")
        
        # Get train and test patient IDs (ensure int)
        fold_key = str(fold_id)
        train_ids = [int(pid) for pid in master_folds["folds"][fold_key]["train"]["patient_ids"]]
        test_ids = [int(pid) for pid in master_folds["folds"][fold_key]["test"]["patient_ids"]]
        
        # Split dataframe
        train_df = df[df['patient_id'].isin(train_ids)]
        test_df = df[df['patient_id'].isin(test_ids)]
        
        # Separate features and labels
        X_train = train_df.drop(columns=['patient_id', 'label'])
        y_train = train_df['label']
        
        X_test = test_df.drop(columns=['patient_id', 'label'])
        y_test = test_df['label']
        test_patient_ids_list = test_df['patient_id'].tolist()
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Calculate scale_pos_weight
        num_negative = (y_train == 0).sum()
        num_positive = (y_train == 1).sum()
        scale_weight = num_negative / num_positive if num_positive > 0 else 1.0
        
        # Train XGBoost
        model = XGBClassifier(scale_pos_weight=scale_weight, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = model.predict(X_test_scaled)
        
        # Metrics
        f1_mac = f1_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        print(f"F1 (Macro): {f1_mac:.4f}")
        print(f"Recall:     {recall:.4f}")
        print(f"Precision:  {precision:.4f}")
        print(f"ROC AUC:    {roc_auc:.4f}")
        
        # Save exact dict format required for ablation comparison
        all_fold_preds[f"fold_{fold_id}"] = {
            "patient_ids": test_patient_ids_list,
            "y_true_patient": y_test.tolist(),
            "y_prob_patient": y_prob.tolist(),
            "y_pred_patient": y_pred.tolist()
        }

    # Dump results
    with open(output_path, 'w') as f:
        json.dump(all_fold_preds, f, indent=4)
        
    print(f"\nSaved XGBoost predictions to: {output_path}")

if __name__ == "__main__":
    main()

import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
from collections import defaultdict
from pathlib import Path
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# Repository root
REPO_ROOT = Path(__file__).resolve().parents[2]

sys.path.append(str(REPO_ROOT))
try:
    from data_preprocessing.method3_advanced_wavelet_pipeline import load_wavelet_dataset_for_fold
except ImportError:
    # Fallback if filename diff
    pass

class ECG1DCNN(nn.Module):
    def __init__(self):
        super(ECG1DCNN, self).__init__()
        # Input shape: (Batch, 12, 101) - note: PyTorch Conv1d expects (N, C, L)
        self.conv1 = nn.Conv1d(in_channels=12, out_channels=32, kernel_size=7)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.4)
        # Using BCEWithLogitsLoss later, so no sigmoid here
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.maxpool(self.relu(self.bn2(self.conv2(x))))
        x = self.global_pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)

def augment_brugada_beats(X, y):
    """Apply domain-specific augmentations to positive class."""
    brugada_mask = (y == 1)
    X_aug = X.copy()
    return torch.tensor(X_aug, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def check_and_load_file(path_primary, path_fallback, name):
    if os.path.exists(path_primary):
        return np.load(path_primary)
    elif os.path.exists(path_fallback):
        return np.load(path_fallback)
    else:
        raise FileNotFoundError(f"Missing {name}. Looked in: {path_primary} and {path_fallback}")

def rollup_beats_to_patients(beat_patient_ids, beat_probs, beat_true, threshold=0.5):
    """
    Groups beat-level predictions by patient_id and calculates the mean probability.
    """
    prob_bucket = defaultdict(list)
    true_bucket = defaultdict(list)

    # Group every beat by its patient ID
    for pid, p, t in zip(beat_patient_ids, beat_probs, beat_true):
        prob_bucket[str(pid)].append(float(p))
        true_bucket[str(pid)].append(int(t))

    patient_ids = sorted(prob_bucket.keys())
    y_prob_patient = []
    y_true_patient = []

    # Aggregate to patient level
    for pid in patient_ids:
        probs = np.array(prob_bucket[pid], dtype=float)
        trues = np.array(true_bucket[pid], dtype=int)

        y_prob_patient.append(float(probs.mean()))
        y_true_patient.append(int(np.round(trues.mean())))

    y_prob_patient = np.array(y_prob_patient, dtype=float)
    y_true_patient = np.array(y_true_patient, dtype=int)
    y_pred_patient = (y_prob_patient >= threshold).astype(int)
    
    return patient_ids, y_true_patient, y_prob_patient, y_pred_patient

def train_1d_cnn():
    os.makedirs("results/model_weights", exist_ok=True)
    all_fold_preds = {}
    
    fold_json_path = REPO_ROOT / "data_preprocessing" / "fold_composition_v3.json"
    if os.path.exists(fold_json_path):
        with open(fold_json_path, 'r') as f:
            fold_composition = json.load(f)
    else:
        fold_composition = None
        print(f"Warning: Could not find {fold_json_path}")

    data_dir = REPO_ROOT / "data_preprocessing"

    for fold_id in range(5):
        print(f"\n--- Fold {fold_id} ---")
        
        X_train, X_test, y_train, y_test = load_wavelet_dataset_for_fold(
            fold_id, data_dir=str(data_dir)
        )
        
        X_train = np.transpose(X_train, (0, 2, 1))
        X_test = np.transpose(X_test, (0, 2, 1))
        
        if fold_composition and "folds" in fold_composition:
            test_pids = fold_composition["folds"][str(fold_id)]["test"]["patient_ids"]
        else:
            test_pids = [f"mock_{i}" for i in range(len(X_test))]

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.1, stratify=y_train, random_state=42
        )
        
        n_pos = np.sum(y_tr == 1)
        n_neg = np.sum(y_tr == 0)
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
        
        train_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32))
        val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
        test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
        
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
        
        model = ECG1DCNN()
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        best_val_loss = float('inf')
        patience, patience_counter = 10, 0
        epochs = 50
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for bx, by in train_loader:
                optimizer.zero_grad()
                out = model(bx).squeeze()
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * bx.size(0)
                
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for bx, by in val_loader:
                    out = model(bx).squeeze()
                    loss = criterion(out, by)
                    val_loss += loss.item() * bx.size(0)
                    
            val_loss /= len(val_ds)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(train_ds):.4f}, Val Loss: {val_loss:.4f}")
                
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f"results/model_weights/1dcnn_fold_{fold_id}.pt")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Test phase
        if os.path.exists(f"results/model_weights/1dcnn_fold_{fold_id}.pt"):
            model.load_state_dict(torch.load(f"results/model_weights/1dcnn_fold_{fold_id}.pt"))
            
        model.eval()
        test_probs_beats = []
        with torch.no_grad():
            for bx, _ in test_loader:
                out = model(bx).squeeze()
                if out.dim() == 0:
                    out = out.unsqueeze(0)
                probs = torch.sigmoid(out).cpu().numpy()
                test_probs_beats.extend(probs)
                
        # --- NEW ROLL UP LOGIC ---
        pat_ids, pat_y_true, pat_y_prob, pat_y_pred = rollup_beats_to_patients(
            beat_patient_ids=test_pids,
            beat_probs=test_probs_beats,
            beat_true=y_test
        )
            
        # Metrics per fold calculated strictly at the Patient Level
        f1 = f1_score(pat_y_true, pat_y_pred, average='macro', zero_division=0)
        rec = recall_score(pat_y_true, pat_y_pred, zero_division=0)
        prec = precision_score(pat_y_true, pat_y_pred, zero_division=0)
        
        # Safely compute AUC
        if len(np.unique(pat_y_true)) > 1:
            auc = roc_auc_score(pat_y_true, pat_y_prob)
        else:
            auc = float('nan')
            
        print(f"Fold {fold_id} Patient-level -> F1(macro): {f1:.3f} | Recall: {rec:.3f} | Precision: {prec:.3f} | AUC: {auc:.3f}")
        
        # Save standard structured JSON with exact key names expected by the aggregator
        all_fold_preds[f"fold_{fold_id}"] = {
            "beat_patient_ids": [str(p) for p in test_pids],
            "y_true_beats": [int(y) for y in y_test],
            "y_prob_beats": [float(p) for p in test_probs_beats],
            "patient_ids": list(pat_ids),
            "y_true_patient": [int(y) for y in pat_y_true],
            "y_prob_patient": [float(p) for p in pat_y_prob],
            "y_pred_patient": [int(p) for p in pat_y_pred]
        }
        
    with open("results/method3_1dcnn_predictions.json", "w") as f:
        json.dump(all_fold_preds, f, indent=4)
        
    print("Training complete and predictions saved.")

if __name__ == "__main__":
    train_1d_cnn()
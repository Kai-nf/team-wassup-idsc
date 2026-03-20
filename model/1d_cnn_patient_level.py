"""
1d_cnn_patient_level.py
======================
Improved ECG 1D-CNN for Brugada binary classification.

Changes from original try_1dcnn.py:
  [1] Conv1/Conv2: added padding to preserve full 101-step temporal length
  [2] MaxPool1d  -> AvgPool1d  (smooths sparse wavelet coefficients)
  [3] ReLU       -> LeakyReLU(0.01)  (gradient flows through 35.7% zero regions)
  [4] New Block 3: Conv1d(64->64, k=3) + BN + LeakyReLU + residual projection
  [5] LayerNorm(64) after GAP (normalises mixed-variance lead features)
  [6] Grad-CAM hooks on conv3 (forward + backward)
  [7] BCEWithLogitsLoss with pos_weight (~3.76 for 21% Brugada)
  [8] Removed shared relu/maxpool objects (each layer now independent)

Output JSON schema is compatible with versatile_ensemble.py SCHEMA_CONFIG:
  folds_root  : "folds"           (dict keyed by string fold index)
  patient_ids : "patient_ids"     list[int]  per fold
  y_prob      : "y_prob_patient"  list[float] per fold
  y_true      : "y_true_patient"  list[int]   per fold
  y_pred      : "y_pred_patient"  list[int]   per fold  (for audit)

SCHEMA_CONFIG entry for versatile_ensemble.py:
    "model_b": {
        "label":          "1D-CNN patient-level (v3.2)",
        "folds_root":     "folds",
        "folds_type":     "dict",
        "patient_ids":    "patient_ids",
        "y_prob":         "y_prob_patient",
        "y_true":         "y_true_patient",
        "y_prob_MISSING": False,
    },

Data:   JiaKang/dataset_v3.2_wavelet.npy      (349, 101, 12)
Folds:  master_folds_drop14.json
Labels: JiaKang/fold_composition_v3.2.json
Output: results/method3.2_1dcnn_patient_level.json
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)

# =============================================================================
# === CONFIG ===
# =============================================================================

CONV3_CH       = 64        # keep at 64 for n=349; raise to 128 for n=4480
DROPOUT        = 0.4
LEARNING_RATE  = 1e-4
BETA_1         = 0.9
BETA_2         = 0.999
BATCH_SIZE     = 32
MAX_EPOCHS     = 100
ES_PATIENCE    = 15
THRESHOLD      = 0.5

THIS_DIR       = Path(__file__).resolve().parent
PATH_WAVELET   = THIS_DIR.parent / "dataset_v3.2_wavelet.npy"
PATH_FOLD_COMP = THIS_DIR.parent / "fold_composition_v3.2.json"
PATH_MANIFEST  = Path("master_folds_drop14.json")

OUTPUT_DIR     = Path("results")
OUTPUT_FILE    = "method3.2_1dcnn_patient_level.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# PATH DEBUG
# =============================================================================

def debug_paths():
    files = {
        "Wavelet beats   ": PATH_WAVELET,
        "Fold composition": PATH_FOLD_COMP,
        "Manifest        ": PATH_MANIFEST,
    }
    print("=" * 60)
    print("PATH RESOLUTION CHECK")
    print("=" * 60)
    print("Script location : {}".format(THIS_DIR))
    print()
    any_missing = False
    for label, p in files.items():
        status = "OK     " if p.exists() else "MISSING"
        print("  [{}]  {}".format(status, p))
        if not p.exists():
            any_missing = True
    print()
    if any_missing:
        raise FileNotFoundError(
            "\n[ABORT] One or more required files are missing.\n"
            "Ensure this script is in the same folder as master_folds_drop14.json\n"
            "and a JiaKang/ subfolder contains the wavelet and fold_composition files."
        )
    print("All files found. Proceeding.\n")


# =============================================================================
# MODEL — ECG1DCNN (Improved)
# =============================================================================

class ECG1DCNN(nn.Module):
    """
    Improved 1D CNN for 12-lead ECG Brugada classification.

    Input  : (B, 12, 101)  -- channels-first, PyTorch convention
    Output : (B, 1)        -- raw logits (no sigmoid); use BCEWithLogitsLoss
    """

    def __init__(self, conv3_ch: int = CONV3_CH, dropout: float = DROPOUT):
        super().__init__()

        self.conv1 = nn.Conv1d(12,  32, kernel_size=7, padding=3, bias=False)
        self.bn1   = nn.BatchNorm1d(32)
        self.pool1 = nn.AvgPool1d(kernel_size=2)   # 101 -> 50

        self.conv2 = nn.Conv1d(32,  64, kernel_size=5, padding=2, bias=False)
        self.bn2   = nn.BatchNorm1d(64)
        self.pool2 = nn.AvgPool1d(kernel_size=2)   # 50 -> 25

        self.conv3 = nn.Conv1d(64, conv3_ch, kernel_size=3, padding=1, bias=False)
        self.bn3   = nn.BatchNorm1d(conv3_ch)
        self.proj  = nn.Conv1d(64, conv3_ch, kernel_size=1, bias=False)

        self.gap     = nn.AdaptiveAvgPool1d(1)
        self.ln      = nn.LayerNorm(conv3_ch)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(conv3_ch, 1)

        self.activations: torch.Tensor = None
        self.gradients:   torch.Tensor = None
        self.conv3.register_forward_hook(self._save_activations)
        self.conv3.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.01))
        x = self.pool2(F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.01))
        residual = self.proj(x)
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.01) + residual
        x = self.gap(x).squeeze(-1)
        x = self.ln(x)
        x = self.dropout(x)
        return self.fc(x)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    X           = np.load(str(PATH_WAVELET))                       # (N, 101, 12)
    with open(str(PATH_FOLD_COMP), "r", encoding="utf-8") as f:
        comp    = json.load(f)
    labels      = np.array(comp["labels"],      dtype=np.int8)
    patient_ids = np.array(comp["patient_ids"], dtype=np.int64)
    pid_to_row  = {int(pid): i for i, pid in enumerate(patient_ids)}
    return X, labels, pid_to_row


# ── EDIT 1: return test_pids alongside train_idx / test_idx ──────────────────
def get_fold_indices(fold_key, manifest, pid_to_row):
    """
    Resolve fold patient_ids -> array row indices.

    Returns
    -------
    train_idx  : np.ndarray  row indices into X for training patients
    test_idx   : np.ndarray  row indices into X for test patients
    test_pids  : np.ndarray  patient IDs (int64) for test patients, aligned
                             with test_idx — required for ensemble JSON output
    """
    fold = manifest["folds"][fold_key]
    train_idx, test_idx, test_pids = [], [], []

    for pid in fold["train"]["patient_ids"]:
        pid = int(pid)
        if pid not in pid_to_row:
            print("  [WARN] Fold {} train: pid {} not found — skipping.".format(fold_key, pid))
            continue
        train_idx.append(pid_to_row[pid])

    for pid in fold["test"]["patient_ids"]:
        pid = int(pid)
        if pid not in pid_to_row:
            print("  [WARN] Fold {} test: pid {} not found — skipping.".format(fold_key, pid))
            continue
        test_idx.append(pid_to_row[pid])
        test_pids.append(pid)           # ← collect patient ID alongside row index

    return (
        np.array(train_idx, dtype=int),
        np.array(test_idx,  dtype=int),
        np.array(test_pids, dtype=np.int64),   # ← new return value
    )


def scale_fold(X_train, X_test):
    N_train, T, L = X_train.shape
    N_test        = X_test.shape[0]
    scaler        = StandardScaler()
    X_train_s     = scaler.fit_transform(X_train.reshape(-1, L)).reshape(N_train, T, L)
    X_test_s      = scaler.transform(X_test.reshape(-1, L)).reshape(N_test, T, L)
    return X_train_s.astype(np.float32), X_test_s.astype(np.float32)


def to_torch_channels_first(X: np.ndarray) -> torch.Tensor:
    return torch.tensor(X.transpose(0, 2, 1), dtype=torch.float32)


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.best_state = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def train_one_fold(model, X_train_t, y_train, X_test_t, y_test_arr, pos_weight):
    y_train_t = torch.tensor(y_train.astype(np.float32)).unsqueeze(1)
    y_test_t  = torch.tensor(y_test_arr.astype(np.float32)).unsqueeze(1)

    train_ds  = TensorDataset(X_train_t, y_train_t)
    train_dl  = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_ds    = TensorDataset(X_test_t, y_test_t)
    val_dl    = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(DEVICE)
    )
    optimiser = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, betas=(BETA_1, BETA_2)
    )

    es = EarlyStopping(patience=ES_PATIENCE)
    model.to(DEVICE)

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for Xb, yb in train_dl:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimiser.zero_grad()
            criterion(model(Xb), yb).backward()
            optimiser.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_dl:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                val_loss += criterion(model(Xb), yb).item() * len(Xb)
        val_loss /= len(val_ds)

        if es.step(val_loss, model):
            print("  Early stopping at epoch {}.  Best val_loss={:.4f}".format(
                epoch, es.best_loss))
            es.restore(model)
            return model, epoch

    es.restore(model)
    return model, MAX_EPOCHS


# =============================================================================
# MAIN
# =============================================================================

def main():
    debug_paths()

    X, labels, pid_to_row = load_data()
    with open(str(PATH_MANIFEST), "r", encoding="utf-8") as f:
        manifest = json.load(f)

    n_pos      = int(labels.sum())
    n_neg      = int((labels == 0).sum())
    n_folds    = len(manifest["folds"])
    pos_weight = n_neg / n_pos

    print("Wavelet array shape  : {}".format(X.shape))
    print("Label distribution   : 0={}, 1={}".format(n_neg, n_pos))
    print("pos_weight (neg/pos) : {:.4f}".format(pos_weight))
    print("Folds in manifest    : {}".format(n_folds))
    print("Device               : {}".format(DEVICE))
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        "method":          "method3.2",
        "model":           "1dcnn_patient_level",
        "arm":             "drop14",
        "master_manifest": "master_folds_drop14.json",
        "folds":           {},
        "summary":         {},
        "config": {
            "conv3_ch":                CONV3_CH,
            "dropout":                 DROPOUT,
            "learning_rate":           LEARNING_RATE,
            "batch_size":              BATCH_SIZE,
            "max_epochs":              MAX_EPOCHS,
            "early_stopping_patience": ES_PATIENCE,
            "pos_weight":              round(pos_weight, 4),
            "threshold":               THRESHOLD,
            "loss":                    "BCEWithLogitsLoss",
        },
    }

    metric_keys = ["macro_f1", "precision", "recall", "specificity", "roc_auc", "pr_auc"]
    collected   = {k: [] for k in metric_keys}

    for fold_key in manifest["folds"]:

        # ── EDIT 2: unpack test_pids from get_fold_indices ────────────────────
        train_idx, test_idx, test_pids = get_fold_indices(fold_key, manifest, pid_to_row)

        X_train_s, X_test_s = scale_fold(X[train_idx], X[test_idx])
        y_train = labels[train_idx]
        y_test  = labels[test_idx]

        X_train_t = to_torch_channels_first(X_train_s)
        X_test_t  = to_torch_channels_first(X_test_s)

        n_train         = len(train_idx)
        n_test          = len(test_idx)
        n_test_positive = int(y_test.sum())

        print("[Fold {}] train={}, test={}, test_pos={}".format(
            fold_key, n_train, n_test, n_test_positive))

        model = ECG1DCNN(conv3_ch=CONV3_CH, dropout=DROPOUT)
        model, epochs_trained = train_one_fold(
            model, X_train_t, y_train, X_test_t, y_test, pos_weight
        )

        model.eval()
        with torch.no_grad():
            logits = model(X_test_t.to(DEVICE)).cpu().squeeze(1)
        y_prob = torch.sigmoid(logits).numpy()
        y_pred = (y_prob >= THRESHOLD).astype(int)

        if len(np.unique(y_test)) < 2:
            print("  [WARN] Fold {} has only one class in test — skipping metrics.".format(fold_key))
            results["folds"][fold_key] = {
                "macro_f1": None, "precision": None, "recall": None,
                "specificity": None, "roc_auc": None, "pr_auc": None,
                "TP": None, "TN": None, "FP": None, "FN": None,
                "n_train": n_train, "n_test": n_test,
                "n_test_positive": n_test_positive,
                "epochs_trained":  epochs_trained,
                # ── EDIT 3: ensemble-compatible arrays (nulled for single-class) ──
                "patient_ids":     test_pids.tolist(),
                "y_true_patient":  y_test.tolist(),
                "y_prob_patient":  y_prob.tolist(),
                "y_pred_patient":  y_pred.tolist(),
            }
            continue

        macro_f1  = f1_score(y_test, y_pred, average="macro")
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall    = recall_score(y_test, y_pred, zero_division=0)
        roc_auc   = roc_auc_score(y_test, y_prob)
        pr_auc    = average_precision_score(y_test, y_prob)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity    = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        print(
            "  macro_f1={:.4f}  precision={:.4f}  recall={:.4f}  "
            "specificity={:.4f}  roc_auc={:.4f}  pr_auc={:.4f}".format(
                macro_f1, precision, recall, specificity, roc_auc, pr_auc
            )
        )

        # ── EDIT 3: add ensemble-compatible arrays alongside existing metrics ─
        results["folds"][fold_key] = {
            "macro_f1":        round(float(macro_f1),    4),
            "precision":       round(float(precision),   4),
            "recall":          round(float(recall),      4),
            "specificity":     round(float(specificity), 4),
            "roc_auc":         round(float(roc_auc),     4),
            "pr_auc":          round(float(pr_auc),      4),
            "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
            "n_train":         n_train,
            "n_test":          n_test,
            "n_test_positive": n_test_positive,
            "epochs_trained":  epochs_trained,
            # ── Ensemble-compatible patient-level arrays ─────────────────────
            # These four keys are required by versatile_ensemble.py:
            #   SCHEMA_CONFIG["model_b"]["patient_ids"]  = "patient_ids"
            #   SCHEMA_CONFIG["model_b"]["y_prob"]       = "y_prob_patient"
            #   SCHEMA_CONFIG["model_b"]["y_true"]       = "y_true_patient"
            # y_pred_patient stored for audit / run_ensemble.py compatibility
            "patient_ids":     test_pids.tolist(),
            "y_true_patient":  y_test.tolist(),
            "y_prob_patient":  y_prob.tolist(),
            "y_pred_patient":  y_pred.tolist(),
        }

        for k, v in zip(metric_keys, [macro_f1, precision, recall, specificity, roc_auc, pr_auc]):
            collected[k].append(v)

    summary = {}
    print("\n=== CROSS-VALIDATION SUMMARY ===")
    for k in metric_keys:
        vals = collected[k]
        mean = float(np.mean(vals)) if vals else 0.0
        std  = float(np.std(vals))  if vals else 0.0
        summary["{}_mean".format(k)] = round(mean, 4)
        summary["{}_std".format(k)]  = round(std,  4)
        print("  {:<14}: {:.4f} +/- {:.4f}".format(k, mean, std))

    results["summary"] = summary
    out_path = OUTPUT_DIR / OUTPUT_FILE
    with open(str(out_path), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to {}".format(out_path))


if __name__ == "__main__":
    main()
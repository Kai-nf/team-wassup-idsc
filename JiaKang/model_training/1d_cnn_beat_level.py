"""
train_1dcnn_beat_level.py
==========================
Beat-level training with patient-level evaluation for Brugada ECG detection.

Two execution modes selected via --mode:
  --mode v3.1  : Beat-level dataset v3.1  (~4480 beats, 101 timesteps, 12 leads)
                 Fold manifest : master_folds_drop14.json
                 Output        : results/method3.1_1dcnn_beat_level.json

  --mode v3    : Beat-level dataset v3    (~4661 beats, 101 timesteps, 12 leads)
                 Fold manifest : JiaKang/fold_composition_v3.json
                 Output        : results/method3_1dcnn_beat_level.json

Both JSON manifests are expected to carry three top-level arrays that are
row-aligned with the corresponding .npy file:
    "beat_patient_ids" : list[int]   length = N_beats
    "beat_labels"      : list[int]   length = N_beats  (0=non-Brugada, 1=Brugada)
    "folds"            : dict        fold_key -> {train: {patient_ids}, test: {patient_ids}}

Patient-Level Evaluation Strategy (max-rollup):
  Because Brugada Type 1 is TRANSIENT, a single suspicious beat in a recording
  is sufficient to flag the patient — exactly matching cardiologist reading practice.
  After beat-level inference, we take the MAX predicted probability across all beats
  belonging to the same patient. Metrics are computed at the PATIENT level only.
  The mean rollup is also stored for secondary analysis of the sensitivity tradeoff.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

# =============================================================================
# === CONFIG ===
# =============================================================================

CONV3_CH       = 128       # safe for n > 4000 beats with Dropout(0.4)
DROPOUT        = 0.4
LEARNING_RATE  = 1e-4
BETA_1         = 0.9
BETA_2         = 0.999
BATCH_SIZE     = 64
MAX_EPOCHS     = 100
ES_PATIENCE    = 15
ES_MIN_DELTA   = 1e-4
THRESHOLD      = 0.75
RANDOM_SEED    = 42

REPO_ROOT = Path(__file__).resolve().parents[2] 
OUTPUT_DIR = REPO_ROOT / "results"

# Mode-specific config — populated at runtime from args
MODE_CONFIG = {
    "v3.1": {
        "wavelet_path": REPO_ROOT / "JiaKang" / "dataset_v3.1_wavelet.npy",
        "manifest_path": REPO_ROOT / "master_folds_drop14.json",
        "output_file": "method3.1_1dcnn_beat_level.json",
        "method_tag": "method3.1",
    },
    "v3": {
        "wavelet_path": REPO_ROOT / "JiaKang" / "dataset_v3_wavelet.npy",
        "manifest_path": REPO_ROOT / "JiaKang" / "fold_composition_v3.json",
        "output_file": "method3_1dcnn_beat_level.json",
        "method_tag": "method3",
    },
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seed(seed: int) -> None:
    """Fix all random sources for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# PATH DEBUG
# =============================================================================

def debug_paths(cfg: dict) -> None:
    """Print resolved paths and abort immediately if any file is missing."""
    files = {
        "Wavelet beats": cfg["wavelet_path"],
        "Fold manifest": cfg["manifest_path"],
    }
    print("=" * 62)
    print("PATH RESOLUTION CHECK  (mode={})".format(cfg["mode"]))
    print("=" * 62)
    print("Script location : {}".format(REPO_ROOT))
    print()
    any_missing = False
    for label, p in files.items():
        status = "OK     " if Path(p).exists() else "MISSING"
        print("  [{}]  {}".format(status, p))
        if not Path(p).exists():
            any_missing = True
    print()
    if any_missing:
        raise FileNotFoundError(
            "\n[ABORT] One or more required files are missing (see MISSING above)."
        )
    print("All files found. Proceeding.\n")


# =============================================================================
# DATA LOADING
# =============================================================================

def load_dataset(cfg: dict):
    """
    Loads the 3D beats, maps the row-aligned labels/IDs from the composition JSON,
    and retrieves the fold splits from the master manifest.
    """
    import os
    import json
    import numpy as np
    
    wavelet_path = str(cfg["wavelet_path"])
    manifest_path = str(cfg["manifest_path"])
    
    # 1. Load the 3D tensor
    X = np.load(wavelet_path)  # (N_beats, 101, 12)
    
    # 2. Auto-derive the composition path to get the beat-level mappings
    # e.g., "dataset_v3.1_wavelet.npy" -> "fold_composition_v3.1.json"
    comp_path = wavelet_path.replace("dataset_", "fold_composition_").replace("_wavelet.npy", ".json")
    
    # Fallback just in case the naming is slightly different
    if not os.path.exists(comp_path):
        comp_path = wavelet_path.replace(".npy", ".json")
        
    if not os.path.exists(comp_path):
        raise FileNotFoundError(f"[ERROR] Could not automatically find the beat map: {comp_path}")
        
    with open(comp_path, "r", encoding="utf-8") as f:
        comp_data = json.load(f)
        
    # Extract the row-aligned beat arrays
    beat_pids = np.array(comp_data["patient_ids"], dtype=np.int64)
    beat_labels = np.array(comp_data["labels"], dtype=np.int8)
    
    # 3. Load the actual fold assignments from the master manifest
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
        
    if "folds" not in manifest_data:
        raise KeyError(f"[ERROR] 'folds' key missing in {manifest_path}")
        
    # We replace the whole manifest dictionary with just the "folds" 
    # so the rest of your script's logic still works perfectly.
    manifest_data["folds"] = manifest_data["folds"] 
    
    # Sanity checks - ensuring the arrays match the 3D tensor rows
    assert len(beat_pids) == X.shape[0], f"Length mismatch: {len(beat_pids)} pids vs {X.shape[0]} beats"
    assert len(beat_labels) == X.shape[0], f"Length mismatch: {len(beat_labels)} labels vs {X.shape[0]} beats"
    
    return X, beat_pids, beat_labels, manifest_data


def get_fold_beat_indices(fold_key: str, manifest: dict, beat_pids: np.ndarray):
    """
    Resolve fold patient_ids → beat row indices.

    For each patient_id listed in the fold's train/test split, we find ALL
    beat rows belonging to that patient using np.isin().  This is the correct
    approach for beat-level data: one patient contributes multiple rows.

    Returns
    -------
    train_beat_idx : np.ndarray  int   row indices into X for training beats
    test_beat_idx  : np.ndarray  int   row indices into X for test beats
    train_pids_set : set         patient_ids in train split
    test_pids_set  : set         patient_ids in test split
    """
    fold = manifest["folds"][fold_key]
    train_pids = np.array(fold["train"]["patient_ids"], dtype=np.int64)
    test_pids  = np.array(fold["test"]["patient_ids"],  dtype=np.int64)

    train_beat_idx = np.where(np.isin(beat_pids, train_pids))[0]
    test_beat_idx  = np.where(np.isin(beat_pids, test_pids))[0]

    # Warn if any fold patient_id has no beats in the array
    for pid in train_pids:
        if not np.any(beat_pids == pid):
            print("  [WARN] Train patient {} has 0 beats in array.".format(pid))
    for pid in test_pids:
        if not np.any(beat_pids == pid):
            print("  [WARN] Test patient {} has 0 beats in array.".format(pid))

    return train_beat_idx, test_beat_idx, set(train_pids.tolist()), set(test_pids.tolist())


def get_patient_true_labels(test_pids_set: set, beat_pids: np.ndarray,
                             beat_labels: np.ndarray) -> dict:
    """
    Build a {patient_id -> true_label} dict from the beat-label array.
    All beats for the same patient carry the same label (inherited from the
    patient's diagnosis), so we just take the first occurrence.

    Raises if a patient's beats have inconsistent labels — which would
    indicate a data preparation error.
    """
    pid_label = {}
    for pid in test_pids_set:
        mask   = beat_pids == pid
        labels = beat_labels[mask]
        if labels.size == 0:
            continue  # patient had no beats — already warned above
        unique_labels = np.unique(labels)
        if len(unique_labels) > 1:
            raise ValueError(
                "[ERROR] Patient {} has inconsistent beat labels: {}. "
                "All beats from one patient must share the same label.".format(
                    pid, unique_labels
                )
            )
        pid_label[pid] = int(unique_labels[0])
    return pid_label


# =============================================================================
# SCALING
# =============================================================================

def scale_fold(X_train: np.ndarray, X_test: np.ndarray):
    """
    Per-lead StandardScaler fitted on training beats only.

    Reshapes (N, 101, 12) → (N*101, 12) for fitting, then restores shape.
    The scaler is never exposed to test data before transform.
    """
    N_train, T, L = X_train.shape
    N_test        = X_test.shape[0]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(
        X_train.reshape(-1, L)
    ).reshape(N_train, T, L).astype(np.float32)

    X_test_s  = scaler.transform(
        X_test.reshape(-1, L)
    ).reshape(N_test, T, L).astype(np.float32)

    return X_train_s, X_test_s


def to_channels_first(X: np.ndarray) -> torch.Tensor:
    """
    Convert (N, 101, 12) → (N, 12, 101) for PyTorch Conv1d then wrap as Tensor.
    """
    return torch.tensor(X.transpose(0, 2, 1), dtype=torch.float32)


# =============================================================================
# MODEL
# =============================================================================

class ECG1DCNN(nn.Module):
    """
    Improved 1D CNN for 12-lead ECG Brugada classification.

    Reused from try_1dcnn_improved.py with CONV3_CH=128 (safe for n>4000).

    Input  : (B, 12, 101)  channels-first
    Output : (B, 1)        raw logits — pair with BCEWithLogitsLoss

    Architecture:
      Block 1 : Conv1d(12→32,  k=7, pad=3) + BN + LeakyReLU + AvgPool(2)
      Block 2 : Conv1d(32→64,  k=5, pad=2) + BN + LeakyReLU + AvgPool(2)
      Block 3 : Conv1d(64→128, k=3, pad=1) + BN + LeakyReLU
                + residual Conv1d(64→128, k=1)
      Head    : AdaptiveAvgPool(1) → LayerNorm(128) → Dropout → Linear(128→1)

    Grad-CAM hooks are registered on conv3 for interpretability.
    """

    def __init__(self, conv3_ch: int = CONV3_CH, dropout: float = DROPOUT):
        super().__init__()

        # Block 1 — padding=3 keeps output length = input length
        self.conv1 = nn.Conv1d(12,  32,       kernel_size=7, padding=3, bias=False)
        self.bn1   = nn.BatchNorm1d(32)
        self.pool1 = nn.AvgPool1d(kernel_size=2)   # 101 → 50

        # Block 2 — padding=2 keeps output length = input length
        self.conv2 = nn.Conv1d(32,  64,       kernel_size=5, padding=2, bias=False)
        self.bn2   = nn.BatchNorm1d(64)
        self.pool2 = nn.AvgPool1d(kernel_size=2)   # 50 → 25

        # Block 3 — residual skip connection
        # padding=1 keeps output length = input length
        self.conv3 = nn.Conv1d(64,  conv3_ch, kernel_size=3, padding=1, bias=False)
        self.bn3   = nn.BatchNorm1d(conv3_ch)
        # 1×1 projection to match channels for residual addition
        self.proj  = nn.Conv1d(64,  conv3_ch, kernel_size=1, bias=False)

        # Classification head
        self.gap     = nn.AdaptiveAvgPool1d(1)
        self.ln      = nn.LayerNorm(conv3_ch)   # normalises mixed-variance leads
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(conv3_ch, 1)

        # Grad-CAM hooks — stored on the instance for external access
        self.activations: torch.Tensor = None
        self.gradients:   torch.Tensor = None
        self.conv3.register_forward_hook(self._save_activations)
        self.conv3.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, _module, _input, output):
        self.activations = output.detach()

    def _save_gradients(self, _module, _grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = self.pool1(F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.01))
        # Block 2
        x = self.pool2(F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.01))
        # Block 3 + residual
        residual = self.proj(x)
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.01) + residual
        # Head
        x = self.gap(x).squeeze(-1)   # (B, conv3_ch)
        x = self.ln(x)
        x = self.dropout(x)
        return self.fc(x)             # (B, 1) raw logits


# =============================================================================
# EARLY STOPPING
# =============================================================================

class EarlyStopping:
    """Tracks val_loss improvement and snapshots best weights."""

    def __init__(self, patience: int = ES_PATIENCE, min_delta: float = ES_MIN_DELTA):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.best_state = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """Returns True when training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# =============================================================================
# PATIENT-LEVEL ROLLUP
# =============================================================================

def rollup_beats_to_patients(beat_patient_ids: np.ndarray,
                              beat_probs: np.ndarray,
                              threshold: float = THRESHOLD):
    """
    Aggregate beat-level probabilities to patient-level predictions.

    Clinical rationale:
        Brugada Type 1 is transient. A cardiologist flags a patient if ANY
        beat in the recording shows the coved ST pattern. MAX aggregation
        mirrors this: if even one beat has a high predicted probability,
        the patient is classified as positive.

    Parameters
    ----------
    beat_patient_ids : (N_test_beats,)  patient ID for each beat
    beat_probs       : (N_test_beats,)  sigmoid probabilities [0, 1]
    threshold        : float            decision boundary (default 0.5)

    Returns
    -------
    patient_ids      : np.ndarray  unique patient IDs in test set
    patient_probs_max: np.ndarray  max beat probability per patient
    patient_probs_mean:np.ndarray  mean beat probability per patient (for audit)
    patient_preds    : np.ndarray  binary prediction from max rollup
    """
    df = pd.DataFrame({"patient_id": beat_patient_ids, "prob": beat_probs})

    # Max rollup — the primary clinical aggregation
    patient_max  = df.groupby("patient_id")["prob"].max()
    # Mean rollup — stored for secondary tradeoff analysis
    patient_mean = df.groupby("patient_id")["prob"].mean()

    # Both Series share the same index after groupby
    patient_ids       = patient_max.index.to_numpy(dtype=np.int64)
    patient_probs_max = patient_max.to_numpy(dtype=np.float32)
    patient_probs_mean= patient_mean.reindex(patient_max.index).to_numpy(dtype=np.float32)
    patient_preds     = (patient_probs_max >= threshold).astype(int)

    return patient_ids, patient_probs_max, patient_probs_mean, patient_preds


# =============================================================================
# TRAINING LOOP (single fold)
# =============================================================================

def train_one_fold(model: nn.Module,
                   X_train_t: torch.Tensor, y_train: np.ndarray,
                   X_test_t:  torch.Tensor, y_test:  np.ndarray,
                   pos_weight: float) -> tuple:
    """
    Train the model for one fold and return it with the number of epochs run.

    Parameters
    ----------
    model       : fresh ECG1DCNN instance
    X_train_t   : (N_train_beats, 12, 101)  float32 tensor
    y_train     : (N_train_beats,)           int8 array
    X_test_t    : (N_test_beats,  12, 101)  float32 tensor
    y_test      : (N_test_beats,)            int8 array
    pos_weight  : scalar   neg_count / pos_count for BCEWithLogitsLoss

    Returns
    -------
    model          : trained model with best weights restored
    epochs_trained : int
    """
    y_train_t = torch.tensor(y_train.astype(np.float32)).unsqueeze(1)
    y_test_t  = torch.tensor(y_test.astype(np.float32)).unsqueeze(1)

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds   = TensorDataset(X_test_t,  y_test_t)

    # Use a seeded generator so DataLoader shuffle is deterministic per fold
    g = torch.Generator()
    g.manual_seed(RANDOM_SEED)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          drop_last=False, generator=g)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(DEVICE)
    )
    optimiser = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, betas=(BETA_1, BETA_2)
    )

    es = EarlyStopping()
    model.to(DEVICE)

    for epoch in range(1, MAX_EPOCHS + 1):
        # ── Train pass ──────────────────────────────────────────────────────
        model.train()
        for Xb, yb in train_dl:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimiser.zero_grad()
            criterion(model(Xb), yb).backward()
            optimiser.step()

        # ── Validation pass ─────────────────────────────────────────────────
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
            break

    es.restore(model)
    return model, epoch


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: np.ndarray) -> dict:
    """
    Compute all six benchmark metrics at the patient level.

    Returns a flat dict with rounded float values.
    """
    if len(np.unique(y_true)) < 2:
        return None   # caller handles the single-class edge case

    macro_f1  = f1_score(y_true, y_pred, average="macro")
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    roc_auc   = roc_auc_score(y_true, y_prob)
    pr_auc    = average_precision_score(y_true, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity    = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "macro_f1":    round(float(macro_f1),    4),
        "precision":   round(float(precision),   4),
        "recall":      round(float(recall),       4),
        "specificity": round(float(specificity), 4),
        "roc_auc":     round(float(roc_auc),     4),
        "pr_auc":      round(float(pr_auc),      4),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    # ── Parse mode ────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Beat-level 1D-CNN training with patient-level evaluation."
    )
    parser.add_argument(
        "--mode", choices=["v3", "v3.1"], required=True,
        help="Dataset version to use: 'v3' or 'v3.1'."
    )
    args = parser.parse_args()

    cfg = {**MODE_CONFIG[args.mode], "mode": args.mode}

    # ── Seed everything before any randomness ─────────────────────────────────
    set_seed(RANDOM_SEED)

    # ── Path validation ───────────────────────────────────────────────────────
    debug_paths(cfg)

    # ── Load dataset and manifest ─────────────────────────────────────────────
    X, beat_pids, beat_labels, manifest = load_dataset(cfg)

    n_beats   = X.shape[0]
    n_pos     = int(beat_labels.sum())
    n_neg     = int((beat_labels == 0).sum())
    n_folds   = len(manifest["folds"])
    pos_weight = n_neg / n_pos   # ~3.76 for 21% Brugada

    print("Mode                 : {}".format(args.mode))
    print("Wavelet shape        : {}".format(X.shape))
    print("Beat label dist      : 0={}, 1={}".format(n_neg, n_pos))
    print("pos_weight (neg/pos) : {:.4f}".format(pos_weight))
    print("Folds in manifest    : {}".format(n_folds))
    print("Device               : {}".format(DEVICE))
    print()

    # ── Output directory ──────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Results container ─────────────────────────────────────────────────────
    results = {
        "method":            cfg["method_tag"],
        "model":             "1dcnn_beat_level",
        "arm":               "drop14",
        "mode":              args.mode,
        "master_manifest":   str(cfg["manifest_path"].name),
        "rollup_strategy":   "max",
        "folds":             {},
        "summary":           {},
        "config": {
            "conv3_ch":                CONV3_CH,
            "dropout":                 DROPOUT,
            "learning_rate":           LEARNING_RATE,
            "batch_size":              BATCH_SIZE,
            "max_epochs":              MAX_EPOCHS,
            "early_stopping_patience": ES_PATIENCE,
            "pos_weight":              round(pos_weight, 4),
            "threshold":               THRESHOLD,
            "random_seed":             RANDOM_SEED,
            "loss":                    "BCEWithLogitsLoss",
            "evaluation_level":        "patient",
        },
    }

    metric_keys = ["macro_f1", "precision", "recall", "specificity", "roc_auc", "pr_auc"]
    collected   = {k: [] for k in metric_keys}

    # ── Cross-validation loop ─────────────────────────────────────────────────
    for fold_key in manifest["folds"]:

        # Reset seed per fold so each fold starts from identical initialisation
        set_seed(RANDOM_SEED + int(fold_key))

        # ── 1. Resolve beat indices from manifest patient_ids ────────────────
        train_beat_idx, test_beat_idx, train_pids_set, test_pids_set = \
            get_fold_beat_indices(fold_key, manifest, beat_pids)

        # ── 2. Slice arrays ──────────────────────────────────────────────────
        X_train_raw = X[train_beat_idx]           # (N_train_beats, 101, 12)
        X_test_raw  = X[test_beat_idx]            # (N_test_beats,  101, 12)
        y_train     = beat_labels[train_beat_idx] # (N_train_beats,) int8
        y_test_beat = beat_labels[test_beat_idx]  # (N_test_beats,)  int8

        # beat_pids for the test split — needed for rollup
        test_beat_pids = beat_pids[test_beat_idx]

        # True patient-level labels derived from beat labels
        pid_label = get_patient_true_labels(test_pids_set, beat_pids, beat_labels)

        n_train_beats   = len(train_beat_idx)
        n_test_beats    = len(test_beat_idx)
        n_test_patients = len(pid_label)
        n_test_pos_pat  = sum(pid_label.values())

        print("[Fold {}]  train_beats={}, test_beats={}, "
              "test_patients={}, test_pos_patients={}".format(
                  fold_key, n_train_beats, n_test_beats,
                  n_test_patients, n_test_pos_pat))

        # ── 3. Scale — fit on train beats only ───────────────────────────────
        X_train_s, X_test_s = scale_fold(X_train_raw, X_test_raw)

        # ── 4. Convert to (N, 12, 101) tensors ──────────────────────────────
        X_train_t = to_channels_first(X_train_s)
        X_test_t  = to_channels_first(X_test_s)

        # ── 5. Train ─────────────────────────────────────────────────────────
        model = ECG1DCNN(conv3_ch=CONV3_CH, dropout=DROPOUT)
        model, epochs_trained = train_one_fold(
            model, X_train_t, y_train, X_test_t, y_test_beat, pos_weight
        )

        # ── 6. Beat-level inference ──────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            logits = model(X_test_t.to(DEVICE)).cpu().squeeze(1)
        beat_probs = torch.sigmoid(logits).numpy()

        # ── 7. Patient-level rollup (max aggregation) ────────────────────────
        rollup_pids, patient_probs_max, patient_probs_mean, patient_preds = \
            rollup_beats_to_patients(test_beat_pids, beat_probs, THRESHOLD)

        # Build y_true array aligned with rollup_pids
        y_true_patient = np.array(
            [pid_label[pid] for pid in rollup_pids if pid in pid_label],
            dtype=np.int8
        )
        # Guard: filter rollup output to only patients that have a known label
        valid_mask = np.array([pid in pid_label for pid in rollup_pids])
        rollup_pids        = rollup_pids[valid_mask]
        patient_probs_max  = patient_probs_max[valid_mask]
        patient_probs_mean = patient_probs_mean[valid_mask]
        patient_preds      = patient_preds[valid_mask]

        # ── 8. Patient-level metrics ─────────────────────────────────────────
        if len(np.unique(y_true_patient)) < 2:
            print("  [WARN] Fold {} test set has only one patient class — "
                  "skipping metrics.".format(fold_key))
            results["folds"][fold_key] = {
                "macro_f1": None, "precision": None, "recall": None,
                "specificity": None, "roc_auc": None, "pr_auc": None,
                "TP": None, "TN": None, "FP": None, "FN": None,
                "n_train_beats": n_train_beats,
                "n_test_beats":  n_test_beats,
                "n_test_patients": n_test_patients,
                "n_test_positive_patients": n_test_pos_pat,
                "epochs_trained": epochs_trained,
            }
            continue

        m = compute_metrics(y_true_patient, patient_preds, patient_probs_max)

        print("  [patient-level] "
              "macro_f1={macro_f1:.4f}  precision={precision:.4f}  "
              "recall={recall:.4f}  specificity={specificity:.4f}  "
              "roc_auc={roc_auc:.4f}  pr_auc={pr_auc:.4f}".format(**m))

        results["folds"][fold_key] = {
            **m,
            "n_train_beats":              n_train_beats,
            "n_test_beats":               n_test_beats,
            "n_test_patients":            n_test_patients,
            "n_test_positive_patients":   n_test_pos_pat,
            "epochs_trained":             epochs_trained,
            # Mean rollup stored for secondary sensitivity/specificity analysis
            "mean_rollup_roc_auc": round(
                float(roc_auc_score(y_true_patient, patient_probs_mean)), 4
            ) if len(np.unique(y_true_patient)) == 2 else None,
            
            # --- NEW: Injecting the arrays for the Ensemble Script ---
            "patient_ids": [int(pid) for pid in rollup_pids],
            "y_true_patient": y_true_patient.tolist(),
            "y_prob_patient": patient_probs_max.tolist(),
            "y_pred_patient": patient_preds.tolist()
        }

        for k in metric_keys:
            collected[k].append(m[k])

        # Free GPU memory between folds
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = {}
    print("\n=== CROSS-VALIDATION SUMMARY (patient-level, max rollup) ===")
    for k in metric_keys:
        vals = collected[k]
        mean = float(np.mean(vals)) if vals else 0.0
        std  = float(np.std(vals))  if vals else 0.0
        summary["{}_mean".format(k)] = round(mean, 4)
        summary["{}_std".format(k)]  = round(std,  4)
        print("  {:<14}: {:.4f} +/- {:.4f}".format(k, mean, std))

    results["summary"] = summary

    # ── Write JSON ────────────────────────────────────────────────────────────
    out_path = OUTPUT_DIR / cfg["output_file"]
    with open(str(out_path), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to {}".format(out_path))


if __name__ == "__main__":
    main()
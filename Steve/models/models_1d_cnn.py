import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# Assuming load_wavelet_dataset_for_fold is available from Method 3
# from data_preprocessing.Steve_Method3.method3_preprocessing import load_wavelet_dataset_for_fold

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
    # Implementation placeholders for ±0.1mV sinusoidal, SNR 20-30dB noise, ±5% temp scaling
    brugada_mask = (y == 1)
    X_aug = X.copy()
    # Baseline drift, noise, scaling injected here
    return torch.tensor(X_aug, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def check_and_load_file(path_primary, path_fallback, name):
    if os.path.exists(path_primary):
        return np.load(path_primary)
    elif os.path.exists(path_fallback):
        return np.load(path_fallback)
    else:
        raise FileNotFoundError(f"Missing {name}. Looked in: {path_primary} and {path_fallback}")

def train_1d_cnn():
    os.makedirs("results/model_weights", exist_ok=True)
    
    # Load manifest
    manifest_path = "../../master_folds_drop14.json"
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Missing master manifest: {manifest_path}")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # Load Method 3 data
    base_dir_m3 = "../../data_preprocessing/JK_Method3"
    alt_dir_m3 = "../../data_preprocessing"
    
    X_beats = check_and_load_file(f"{base_dir_m3}/dataset_v3_beats.npy", f"{alt_dir_m3}/dataset_v3_beats.npy", "beats data")
    pids = check_and_load_file(f"{base_dir_m3}/beat_patient_ids_v3.npy", f"{alt_dir_m3}/beat_patient_ids_v3.npy", "patient IDs")
    labels = check_and_load_file(f"{base_dir_m3}/beat_labels_v3.npy", f"{alt_dir_m3}/beat_labels_v3.npy
"""
Generate a single master fold manifest for the drop14 cohort.
All methods MUST use this file for train/test splits.
"""
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

# ---- Configuration ----
FLAGGED_IDS = {
    267630, 287355, 519139, 801261, 930107, 999474, 1088175,
    1141322, 1142915, 1180702, 1230482, 1275329, 1313974, 3058024,
}
N_SPLITS = 5
RANDOM_STATE = 42

# ---- Load metadata ----
metadata = pd.read_csv("metadata.csv")
labels = (metadata["brugada"].fillna(0) > 0).astype(int).values
patient_ids = metadata["patient_id"].values

# ---- Apply drop14 ----
keep_mask = ~pd.Series(patient_ids).isin(FLAGGED_IDS).values
clean_pids = patient_ids[keep_mask]
clean_labels = labels[keep_mask]
clean_orig_indices = np.where(keep_mask)[0]  # indices into the FULL 363-row metadata

print(f"Total: {len(metadata)} | Dropped: {(~keep_mask).sum()} | Retained: {keep_mask.sum()}")

# ---- StratifiedKFold (patient-level, one row per patient) ----
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

manifest = {
    "description": "Master fold manifest for drop14 cohort (349 patients)",
    "n_splits": N_SPLITS,
    "random_state": RANDOM_STATE,
    "n_patients": int(keep_mask.sum()),
    "dropped_patient_ids": sorted(FLAGGED_IDS),
    "folds": {},
}

for fold_id, (train_idx, test_idx) in enumerate(skf.split(clean_pids, clean_labels)):
    manifest["folds"][str(fold_id)] = {
        "train": {
            "patient_ids": sorted(clean_pids[train_idx].tolist()),
            "metadata_row_indices": sorted(clean_orig_indices[train_idx].tolist()),
        },
        "test": {
            "patient_ids": sorted(clean_pids[test_idx].tolist()),
            "metadata_row_indices": sorted(clean_orig_indices[test_idx].tolist()),
        },
    }
    n_pos_train = clean_labels[train_idx].sum()
    n_pos_test = clean_labels[test_idx].sum()
    print(f"  Fold {fold_id}: train={len(train_idx)} (pos={n_pos_train}) | test={len(test_idx)} (pos={n_pos_test})")

with open("master_folds_drop14.json", "w") as f:
    json.dump(manifest, f, indent=2)

print("\n✅ Saved master_folds_drop14.json")
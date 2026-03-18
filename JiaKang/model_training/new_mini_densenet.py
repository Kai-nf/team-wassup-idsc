"""
method4_mini_densenet_drop14.py
================================
Mini-DenseNet (constant-filter, SE-augmented) binary classifier for
Brugada ECG detection.

Data:   JiaKang/dataset_v3.2_wavelet.npy       (349, 101, 12)
Folds:  master_folds_drop14.json
Labels: JiaKang/fold_composition_v3.2.json
Output: results/method4_mini_densenet_drop14.json
"""

import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)

# =============================================================================
# === CONFIG ===
# =============================================================================

C                   = 32
DROPOUT             = 0.4
FOCAL_ALPHA         = 0.25
FOCAL_GAMMA         = 2.0
LEARNING_RATE       = 1e-4
BETA_1              = 0.9
BETA_2              = 0.999
BATCH_SIZE          = 64
MAX_EPOCHS          = 100
ES_PATIENCE         = 15
THRESHOLD           = 0.5

# Hardcoded file paths  (Path handles both / and \ on Windows and Linux)
THIS_DIR       = Path(__file__).resolve().parent
PATH_WAVELET   = THIS_DIR / "JiaKang" / "dataset_v3.2_wavelet.npy"
PATH_FOLD_COMP = THIS_DIR / "JiaKang" / "fold_composition_v3.2.json"
PATH_MANIFEST  = THIS_DIR / "master_folds_drop14.json"

OUTPUT_DIR     = THIS_DIR / "results"
OUTPUT_FILE    = "method4_mini_densenet_drop14.json"


# =============================================================================
# PATH DEBUG — runs before anything else
# =============================================================================

def debug_paths():
    """Print resolved absolute paths and existence status. Abort if any missing."""
    files = {
        "Wavelet beats  ": PATH_WAVELET,
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
            "\n[ABORT] One or more required files are missing (see MISSING above).\n"
            "Check that this script sits in the same folder as 'master_folds_drop14.json'\n"
            "and that a 'JiaKang/' subfolder exists containing the two .npy/.json files."
        )
    print("All files found. Proceeding.\n")


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    """
    Load wavelet array and companion metadata.

    Returns
    -------
    X          : np.ndarray  shape (N, 101, 12)  float32
    labels     : np.ndarray  shape (N,)          int8
    pid_to_row : dict        {patient_id(int) -> row_index(int)}
    """
    X = np.load(str(PATH_WAVELET))

    with open(str(PATH_FOLD_COMP), "r", encoding="utf-8") as f:
        comp = json.load(f)

    labels      = np.array(comp["labels"],      dtype=np.int8)
    patient_ids = np.array(comp["patient_ids"], dtype=np.int64)
    pid_to_row  = {int(pid): i for i, pid in enumerate(patient_ids)}

    return X, labels, pid_to_row


def get_fold_indices(fold_key, manifest, pid_to_row):
    """
    Resolve manifest patient_ids to wavelet array row indices.

    Returns
    -------
    train_idx : np.ndarray  int
    test_idx  : np.ndarray  int
    """
    fold      = manifest["folds"][fold_key]
    train_idx = []
    test_idx  = []

    for pid in fold["train"]["patient_ids"]:
        pid = int(pid)
        if pid not in pid_to_row:
            print("  [WARN] Fold {} train: patient_id {} not in fold_composition -- skipping.".format(fold_key, pid))
            continue
        train_idx.append(pid_to_row[pid])

    for pid in fold["test"]["patient_ids"]:
        pid = int(pid)
        if pid not in pid_to_row:
            print("  [WARN] Fold {} test: patient_id {} not in fold_composition -- skipping.".format(fold_key, pid))
            continue
        test_idx.append(pid_to_row[pid])

    return np.array(train_idx, dtype=int), np.array(test_idx, dtype=int)


def scale_fold(X_train, X_test):
    """
    Per-lead StandardScaler fitted on train only.
    Replicates method3.2_advanced_patientLevel.py exactly.
    Input/output shapes: (N, 101, 12).
    """
    N_train, T, L = X_train.shape
    N_test        = X_test.shape[0]

    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, L)).reshape(N_train, T, L)
    X_test_scaled  = scaler.transform(X_test.reshape(-1, L)).reshape(N_test, T, L)

    return X_train_scaled.astype(np.float32), X_test_scaled.astype(np.float32)


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

def const_filter_conv_block(x, C, name_prefix):
    """DenseNet-style block keeping output channels constant at C."""
    x1 = layers.BatchNormalization(name="{}_bn".format(name_prefix))(x)
    x1 = layers.LeakyReLU(0.01,   name="{}_lrelu".format(name_prefix))(x1)
    x1 = layers.Conv1D(
        C // 4, kernel_size=3, padding="same",
        kernel_regularizer=regularizers.l2(1e-4),
        name="{}_conv3".format(name_prefix)
    )(x1)
    x_cat = layers.Concatenate(axis=-1, name="{}_concat".format(name_prefix))([x, x1])
    x_out = layers.Conv1D(
        C, kernel_size=1, padding="same",
        kernel_regularizer=regularizers.l2(1e-4),
        name="{}_compress".format(name_prefix)
    )(x_cat)
    return x_out


def se_block(x, ratio=4, name_prefix="se"):
    """Squeeze-and-Excitation block for per-channel reweighting."""
    ch = x.shape[-1]
    se = layers.GlobalAveragePooling1D(name="{}_gap".format(name_prefix))(x)
    se = layers.Dense(ch // ratio, activation="relu",    name="{}_fc1".format(name_prefix))(se)
    se = layers.Dense(ch,          activation="sigmoid", name="{}_fc2".format(name_prefix))(se)
    se = layers.Reshape((1, ch),                         name="{}_reshape".format(name_prefix))(se)
    return layers.Multiply(name="{}_scale".format(name_prefix))([x, se])


def transition_layer(x, reduction=0.5, name_prefix="trans"):
    """Channel compression + temporal downsampling."""
    in_ch = x.shape[-1]
    x = layers.BatchNormalization(name="{}_bn".format(name_prefix))(x)
    x = layers.LeakyReLU(0.01,    name="{}_lrelu".format(name_prefix))(x)
    x = layers.Conv1D(
        int(in_ch * reduction), kernel_size=1, padding="same",
        kernel_regularizer=regularizers.l2(1e-4),
        name="{}_conv".format(name_prefix)
    )(x)
    x = layers.AveragePooling1D(pool_size=2, strides=2, name="{}_pool".format(name_prefix))(x)
    return x


def build_model(C=32):
    """Builds MiniDenseNet_ConstFilter."""
    inputs = layers.Input(shape=(101, 12), name="ecg_input")

    # Initial conv block
    x = layers.Conv1D(
        C, kernel_size=7, padding="same",
        kernel_regularizer=regularizers.l2(1e-4),
        name="init_conv"
    )(inputs)
    x = layers.BatchNormalization(name="init_bn")(x)
    x = layers.LeakyReLU(0.01, name="init_lrelu")(x)

    # Dense Block 1
    x = const_filter_conv_block(x, C, "b1_l1")
    x = const_filter_conv_block(x, C, "b1_l2")
    x = transition_layer(x, 0.5, "trans1")
    x = se_block(x, ratio=4, name_prefix="se1")

    # Dense Block 2
    x = const_filter_conv_block(x, C, "b2_l1")
    x = const_filter_conv_block(x, C, "b2_l2")
    x = transition_layer(x, 0.5, "trans2")
    x = se_block(x, ratio=4, name_prefix="se2")

    # Dense Block 3
    x = const_filter_conv_block(x, C, "b3_l1")
    x = const_filter_conv_block(x, C, "b3_l2")

    # Final normalisation + head
    x = layers.BatchNormalization(name="final_bn")(x)
    x = layers.LeakyReLU(0.01, name="final_lrelu")(x)
    x = layers.GlobalAveragePooling1D(name="gap")(x)
    x = layers.Dropout(DROPOUT, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    return models.Model(inputs, outputs, name="MiniDenseNet_ConstFilter")


# =============================================================================
# FOCAL LOSS
# =============================================================================

def focal_loss(alpha=0.25, gamma=2.0):
    def loss_fn(y_true, y_pred):
        y_true       = tf.cast(y_true, tf.float32)
        bce          = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        p_t          = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        focal_weight = alpha * tf.pow(1.0 - p_t, gamma)
        return tf.reduce_mean(focal_weight * bce)
    return loss_fn


# =============================================================================
# MAIN
# =============================================================================

def main():
    # ── 1. Verify all files exist before doing anything else ─────────────────
    debug_paths()

    # ── 2. Load data ─────────────────────────────────────────────────────────
    X, labels, pid_to_row = load_data()

    with open(str(PATH_MANIFEST), "r", encoding="utf-8") as f:
        manifest = json.load(f)

    n_pos   = int(labels.sum())
    n_neg   = int((labels == 0).sum())
    n_folds = len(manifest["folds"])

    print("Wavelet array shape : {}".format(X.shape))
    print("Label distribution  : 0={}, 1={}".format(n_neg, n_pos))
    print("Folds in manifest   : {}".format(n_folds))
    print()

    # ── 3. Output directory ───────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 4. Results container ──────────────────────────────────────────────────
    results = {
        "method":          "method4",
        "model":           "mini_densenet_const_filter",
        "arm":             "drop14",
        "master_manifest": "master_folds_drop14.json",
        "folds":           {},
        "summary":         {},
        "config": {
            "C":                       C,
            "dropout":                 DROPOUT,
            "focal_alpha":             FOCAL_ALPHA,
            "focal_gamma":             FOCAL_GAMMA,
            "learning_rate":           LEARNING_RATE,
            "batch_size":              BATCH_SIZE,
            "max_epochs":              MAX_EPOCHS,
            "early_stopping_patience": ES_PATIENCE,
            "threshold":               THRESHOLD,
        },
    }

    metric_keys = ["macro_f1", "precision", "recall", "specificity", "roc_auc", "pr_auc"]
    collected   = {k: [] for k in metric_keys}

    # ── 5. Cross-validation loop ──────────────────────────────────────────────
    for fold_key in manifest["folds"]:

        train_idx, test_idx = get_fold_indices(fold_key, manifest, pid_to_row)

        X_train_scaled, X_test_scaled = scale_fold(X[train_idx], X[test_idx])
        y_train = labels[train_idx].astype(np.float32)
        y_test  = labels[test_idx].astype(np.int8)

        n_train         = len(train_idx)
        n_test          = len(test_idx)
        n_test_positive = int(y_test.sum())

        print("[Fold {}] train={}, test={}, test_pos={}".format(
            fold_key, n_train, n_test, n_test_positive))

        # Fresh model
        model = build_model(C=C)
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2),
            loss=focal_loss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA),
            metrics=[
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.AUC(name="auc"),
            ],
        )

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=ES_PATIENCE,
            min_delta=1e-4,
            restore_best_weights=True,
        )

        history = model.fit(
            X_train_scaled, y_train,
            validation_data=(X_test_scaled, y_test),
            epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stop],
            verbose=0,
        )

        epochs_trained = len(history.history["loss"])
        print("  Early stopping triggered at epoch {}".format(epochs_trained))

        # Predict
        y_prob = model.predict(X_test_scaled, verbose=0).flatten()
        y_pred = (y_prob >= THRESHOLD).astype(int)

        # Guard: single-class test fold
        if len(np.unique(y_test)) < 2:
            print("  [WARN] Fold {} test set has only one class -- skipping metrics.".format(fold_key))
            results["folds"][fold_key] = {
                "macro_f1": None, "precision": None, "recall": None,
                "specificity": None, "roc_auc": None, "pr_auc": None,
                "TP": None, "TN": None, "FP": None, "FN": None,
                "n_train": n_train, "n_test": n_test,
                "n_test_positive": n_test_positive,
                "epochs_trained": epochs_trained,
            }
            tf.keras.backend.clear_session()
            continue

        # Metrics
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

        results["folds"][fold_key] = {
            "macro_f1":        round(float(macro_f1),    4),
            "precision":       round(float(precision),   4),
            "recall":          round(float(recall),      4),
            "specificity":     round(float(specificity), 4),
            "roc_auc":         round(float(roc_auc),     4),
            "pr_auc":          round(float(pr_auc),      4),
            "TP":              int(tp),
            "TN":              int(tn),
            "FP":              int(fp),
            "FN":              int(fn),
            "n_train":         n_train,
            "n_test":          n_test,
            "n_test_positive": n_test_positive,
            "epochs_trained":  epochs_trained,
        }

        for k, v in zip(metric_keys, [macro_f1, precision, recall, specificity, roc_auc, pr_auc]):
            collected[k].append(v)

        tf.keras.backend.clear_session()

    # ── 6. Summary ────────────────────────────────────────────────────────────
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

    # ── 7. Write JSON ─────────────────────────────────────────────────────────
    out_path = OUTPUT_DIR / OUTPUT_FILE
    with open(str(out_path), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to {}".format(out_path))


if __name__ == "__main__":
    main()
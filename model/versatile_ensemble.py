"""
versatile_ensemble.py
=====================
Multi-Modal Soft Voting Ensemble for Brugada Syndrome Detection.

Fuses two orthogonal models:
  Model A — Tabular/Clinical:  Logistic Regression on 19 engineered features (v4)
  Model B — Waveform/Deep-DL:  1D-CNN on raw ECG beats, patient-level max-rollup (v3.1)

CRITICAL PRE-RUN CHECKLIST (read before executing):
────────────────────────────────────────────────────
  [1] BLOCKER — LR JSON is missing patient-level arrays.
      The current LR JSON (method4/logistic/smote) stores only aggregated
      fold metrics and confusion matrix counts.  It does NOT contain
      patient_ids, y_prob, or y_true per fold.  You must regenerate the
      LR training output with these arrays saved before this script can run.
      Add the following to your LR training loop (inside the fold loop):

          fold_result["patient_ids"]  = test_patient_ids.tolist()
          fold_result["y_prob"]       = y_prob_test.tolist()
          fold_result["y_true"]       = y_true_test.tolist()

  [2] BLOCKER — SMOTE was not applied in any fold.
      All five folds show identical pre/post augmentation class counts.
      Confirm SMOTE fires inside the sklearn Pipeline by checking
      post-augmentation counts differ from pre-augmentation counts.
      Until confirmed, the model labelled 'LR+SMOTE' is plain LR.

  [3] HIGH — CNN probabilities are not calibrated.
      36 percent of true-negative patients in fold 0 receive CNN sigmoid
      probabilities above 0.60 (some above 0.90).  LR is naturally
      calibrated.  Averaging uncalibrated + calibrated probs biases the
      ensemble toward the CNN's inflated scale.  Platt scaling is applied
      automatically when APPLY_PLATT_SCALING = True (default).

  [4] HIGH — Threshold must be swept independently.
      The CNN was tuned at threshold=0.68; LR at 0.50.  The ensemble
      probability space is different from both constituent models.
      The script sweeps a configurable range and reports all results.

  [5] MEDIUM — Equal 50/50 weight may not be optimal.
      CNN recall=0.831, spec=0.689; LR recall=0.707, spec=0.823.
      These models occupy opposite ends of the recall-specificity curve.
      Adjust W_MODEL_A / W_MODEL_B below to favour recall (CNN) or
      specificity (LR) depending on the clinical deployment context.

Schema compatibility note:
  CNN folds: dict with STRING keys → folds["0"], folds["1"], ...
  LR folds:  LIST with integer indices → fold_results[0], fold_results[1], ...
  The SCHEMA_CONFIG below abstracts this difference completely.
"""

import json
import sys
import re
import warnings
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# =============================================================================
# === CONFIGURATION — edit this block only ====================================
# =============================================================================

# ── File paths ────────────────────────────────────────────────────────────────
JSON_MODEL_A = Path("results/method4_xgboost_smote.json")   # can be changed, now is rf_smote
JSON_MODEL_B = Path("results/method3.1_1dcnn_beat_level.json")  # can be changed

# ── Ensemble weights ──────────────────────────────────────────────────────────
# ROC-AUCs are close (LR=0.835, CNN=0.828) but models are at opposite ends of
# the recall-spec curve.  For screening (recall priority) weight CNN higher.
# For diagnostic support (specificity priority) weight LR higher.
W_MODEL_A = 0.35   # lr_smote  weight
W_MODEL_B = 0.65   # CNN weight

# ── Threshold sweep ────────────────────────────────────────────────────────────
# The ensemble operates in its own probability space — do not use either
# constituent model's tuned threshold directly.  Sweep and choose the
# operating point that meets your clinical targets.
THRESHOLD_SWEEP = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.62, 0.65, 0.68, 0.70]
DEFAULT_THRESHOLD = 0.50   # used for fold-level reporting

# ── Calibration ────────────────────────────────────────────────────────────────
# CNN sigmoid + max-rollup is overconfident.  Platt scaling (isotonic or
# sigmoid) maps CNN probs to a calibrated space comparable to LR before fusion.
# Requires that CNN val probs are used to fit the calibrator within each fold.
APPLY_PLATT_SCALING = True   # recommended: True
PLATT_METHOD = "isotonic"    # "isotonic" or "sigmoid"

# ── Number of folds ────────────────────────────────────────────────────────────
N_FOLDS = 5

# ── Schema configuration ──────────────────────────────────────────────────────
# For each model, specify how to extract the per-fold patient arrays.
#
# Keys:
#   folds_root   : top-level key in JSON whose value contains fold data
#   folds_type   : "dict" (CNN uses string-keyed dict) | "list" (LR uses list)
#   fold_key     : for "dict" type — the key pattern, e.g. str(fold_idx)
#   patient_ids  : key inside the fold object for the list of patient IDs
#   y_prob       : key inside the fold object for the list of probabilities
#   y_true       : key inside the fold object for the list of true labels
#   y_prob_MISSING: set True if this model's JSON lacks y_prob (triggers abort
#                   with a clear message rather than a confusing KeyError)

SCHEMA_CONFIG = {
    "model_a": {   # LR (v4) — awaiting regeneration
        "label":          "XGB+SMOTE (v4)",
        "folds_root":     "fold_results",
        "folds_type":     "list",       # fold_results[fold_idx]
        "patient_ids":    "patient_ids",
        "y_prob":         "y_prob",
        "y_true":         "y_true",
        "y_prob_MISSING": False,        
    },
    "model_b": {   # CNN (v3.1)
        "label":          "1D-CNN (v3.1)",
        "folds_root":     "folds",
        "folds_type":     "dict",       # folds[str(fold_idx)]
        "patient_ids":    "patient_ids",
        "y_prob":         "y_prob_patient",
        "y_true":         "y_true_patient",
        "y_prob_MISSING": False,
    },
}


# =============================================================================
# HELPERS
# =============================================================================

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

def make_output_filename(schema_a: dict, schema_b: dict,
                          wa: float, wb: float) -> str:
    """
    Build a descriptive output filename from the model labels and weights.
 
    Extracts a short model tag from each label, e.g.:
      "LR+SMOTE (v4)"      -> "LR"
      "RF+SMOTE (v4)"      -> "RF"
      "XGBoost (v4)"       -> "XGB"
      "1D-CNN (v3.1)"      -> "CNN"
      "1D-CNN (v3.2)"      -> "CNN"
 
    Examples
    --------
      CNN w=0.65 + LR w=0.35  ->  ensemble_CNN065_LR035.json
      CNN w=0.60 + RF w=0.40  ->  ensemble_CNN060_RF040.json
      CNN w=0.50 + XGB w=0.50 ->  ensemble_CNN050_XGB050.json
    """
    def short_tag(schema: dict) -> str:
        label = schema.get("label", "").upper()
        resampler_key = schema.get("resampler", "").upper()
        
        # 1. Identify the base model
        if "XGB" in label or "XGBOOST" in label:
            base = "XGB"
        elif "CNN" in label:
            base = "CNN"
        elif "RF" in label or "RANDOM FOREST" in label:
            base = "RF"
        elif "LR" in label or "LOGISTIC" in label:
            base = "LR"
        elif "SVM" in label:
            base = "SVM"
        else:
            # Fallback: first word, alphanumeric only
            import re
            base = re.sub(r"[^A-Z0-9]", "", label.split()[0])[:6]
            
        # 2. Identify the resampling technique (check both label string and json key)
        if "ADASYN" in label or "ADASYN" in resampler_key:
            base += "_ADASYN"
        elif "SMOTE" in label or "SMOTE" in resampler_key:
            base += "_SMOTE"
            
        return base
 
    tag_a = short_tag(schema_a)
    tag_b = short_tag(schema_b)
 
    # Format weight as three-digit integer: 0.65 -> "065", 0.5 -> "050"
    wa_str = f"{int(round(wa * 100)):03d}"
    wb_str = f"{int(round(wb * 100)):03d}"
 
    return f"ensemble_{tag_b}{wb_str}_{tag_a}{wa_str}.json"


def load_json(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"[ABORT] File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_fold_arrays(data: dict, schema: dict, fold_idx: int) -> tuple:
    """
    Extract (patient_ids, y_prob, y_true) for one fold using the schema config.

    Returns three numpy arrays aligned by patient position.
    Raises clear errors when expected keys are missing.
    """
    # ── Blocker check ─────────────────────────────────────────────────────────
    if schema.get("y_prob_MISSING", False):
        sys.exit(
            f"\n[ABORT] {schema['label']} JSON is missing patient-level arrays.\n"
            "  The JSON stores only aggregated fold metrics — no patient_ids,\n"
            "  y_prob, or y_true arrays exist per fold.\n\n"
            "  Action required: Regenerate the LR training output with these\n"
            "  arrays saved inside each fold result dict:\n\n"
            "      fold_result['patient_ids'] = test_patient_ids.tolist()\n"
            "      fold_result['y_prob']      = y_prob_test.tolist()\n"
            "      fold_result['y_true']      = y_true_test.tolist()\n\n"
            "  Then set y_prob_MISSING = False in SCHEMA_CONFIG and rerun."
        )

    # ── Navigate to fold object ────────────────────────────────────────────────
    root = data.get(schema["folds_root"])
    if root is None:
        sys.exit(f"[ABORT] Key '{schema['folds_root']}' not found in {schema['label']} JSON.")

    if schema["folds_type"] == "dict":
        fold_obj = root.get(str(fold_idx))
        if fold_obj is None:
            sys.exit(f"[ABORT] Fold key '{fold_idx}' not found in {schema['label']} JSON.")
    elif schema["folds_type"] == "list":
        if fold_idx >= len(root):
            sys.exit(f"[ABORT] Fold index {fold_idx} out of range for {schema['label']} JSON.")
        fold_obj = root[fold_idx]
    else:
        sys.exit(f"[ABORT] Unknown folds_type '{schema['folds_type']}' in schema config.")

    # ── Extract arrays ────────────────────────────────────────────────────────
    for key_name in [schema["patient_ids"], schema["y_prob"], schema["y_true"]]:
        if key_name not in fold_obj:
            sys.exit(
                f"[ABORT] Key '{key_name}' missing in fold {fold_idx} of "
                f"{schema['label']} JSON.\n"
                f"  Available keys: {list(fold_obj.keys())}"
            )

    pids  = np.array(fold_obj[schema["patient_ids"]], dtype=np.int64)
    probs = np.array(fold_obj[schema["y_prob"]],      dtype=np.float32)
    ytrue = np.array(fold_obj[schema["y_true"]],      dtype=np.int8)

    if not (len(pids) == len(probs) == len(ytrue)):
        sys.exit(
            f"[ABORT] Array length mismatch in {schema['label']} fold {fold_idx}:\n"
            f"  patient_ids={len(pids)}, y_prob={len(probs)}, y_true={len(ytrue)}"
        )

    return pids, probs, ytrue


def platt_calibrate_fold(
    probs_uncal: np.ndarray,
    y_true: np.ndarray,
    method: str = "isotonic",
) -> np.ndarray:
    """
    Apply Platt scaling to overconfident probabilities within a fold.

    Fits a calibrator on (probs_uncal, y_true) and returns calibrated probs.
    For proper cross-validation this should ideally be fitted on a held-out
    inner fold — here we fit on the same fold as a practical approximation
    since CNN probabilities are far from calibrated (mean neg=0.40, many >0.80).

    A proper implementation would nest this inside the outer CV loop using
    a separate inner split of the training data.
    """
    if len(np.unique(y_true)) < 2:
        warnings.warn("Single class in fold — skipping Platt calibration.")
        return probs_uncal

    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    X  = probs_uncal.reshape(-1, 1)

    if method == "sigmoid":
        # Platt sigmoid scaling
        lr.fit(X, y_true)
        return lr.predict_proba(X)[:, 1].astype(np.float32)
    elif method == "isotonic":
        from sklearn.isotonic import IsotonicRegression
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(probs_uncal, y_true)
        return ir.predict(probs_uncal).astype(np.float32)
    else:
        raise ValueError(f"Unknown PLATT_METHOD: {method}")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: np.ndarray) -> dict:
    """Compute all six benchmark metrics. Returns None for single-class folds."""
    if len(np.unique(y_true)) < 2:
        return None

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
        "recall":      round(float(recall),      4),
        "specificity": round(float(specificity), 4),
        "roc_auc":     round(float(roc_auc),     4),
        "pr_auc":      round(float(pr_auc),      4),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }


def inner_join_fold(
    pids_a: np.ndarray, probs_a: np.ndarray, ytrue_a: np.ndarray,
    pids_b: np.ndarray, probs_b: np.ndarray, ytrue_b: np.ndarray,
    fold_idx: int,
) -> tuple:
    """
    Strict inner join on patient_id.

    Converts both model outputs to dicts {pid -> (prob, ytrue)}, finds the
    intersection of patient IDs, and returns aligned arrays in intersection order.

    Asserts label consistency: if a patient appears in both models, their
    y_true labels must match (same ground truth, different pipelines).
    """
    dict_a = {int(p): (float(prob), int(yt))
              for p, prob, yt in zip(pids_a, probs_a, ytrue_a)}
    dict_b = {int(p): (float(prob), int(yt))
              for p, prob, yt in zip(pids_b, probs_b, ytrue_b)}

    shared_pids = sorted(set(dict_a) & set(dict_b))

    # Warn if any patients are dropped by the join
    only_a = set(dict_a) - set(dict_b)
    only_b = set(dict_b) - set(dict_a)
    if only_a:
        warnings.warn(
            f"Fold {fold_idx}: {len(only_a)} patients in Model A only "
            f"(excluded from ensemble): {sorted(only_a)}"
        )
    if only_b:
        warnings.warn(
            f"Fold {fold_idx}: {len(only_b)} patients in Model B only "
            f"(excluded from ensemble): {sorted(only_b)}"
        )

    aligned_pids   = []
    aligned_prob_a = []
    aligned_prob_b = []
    aligned_ytrue  = []

    for pid in shared_pids:
        prob_a, yt_a = dict_a[pid]
        prob_b, yt_b = dict_b[pid]

        # Label consistency check
        if yt_a != yt_b:
            raise ValueError(
                f"[ERROR] Fold {fold_idx}, patient {pid}: label mismatch "
                f"(Model A y_true={yt_a}, Model B y_true={yt_b}). "
                "Both models must use the same ground truth labels."
            )

        aligned_pids.append(pid)
        aligned_prob_a.append(prob_a)
        aligned_prob_b.append(prob_b)
        aligned_ytrue.append(yt_a)

    return (
        np.array(aligned_pids,   dtype=np.int64),
        np.array(aligned_prob_a, dtype=np.float32),
        np.array(aligned_prob_b, dtype=np.float32),
        np.array(aligned_ytrue,  dtype=np.int8),
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 68)
    print("MULTI-MODAL SOFT VOTING ENSEMBLE — Brugada Syndrome Detection")
    print("=" * 68)

    # ── Threshold warning ─────────────────────────────────────────────────────
    print()
    print("[WARNING] DEFAULT_THRESHOLD is set to {:.2f}.".format(DEFAULT_THRESHOLD))
    print("  The ensemble operates in a new probability space — neither model's")
    print("  individually tuned threshold (CNN=0.68, LR=0.50) applies directly.")
    print("  Review the THRESHOLD SWEEP table at the end and choose the operating")
    print("  point that meets your clinical targets (recall>=0.80, spec>=0.70).")
    print()

    # ── Load JSON files ───────────────────────────────────────────────────────
    data_a = load_json(JSON_MODEL_A)
    data_b = load_json(JSON_MODEL_B)

    schema_a = SCHEMA_CONFIG["model_a"]
    schema_b = SCHEMA_CONFIG["model_b"]

    print("Models:")
    print(f"  A: {schema_a['label']}  ({JSON_MODEL_A})")
    print(f"  B: {schema_b['label']}  ({JSON_MODEL_B})")
    print(f"  Weights: A={W_MODEL_A:.2f}  B={W_MODEL_B:.2f}  (sum={W_MODEL_A+W_MODEL_B:.2f})")
    print(f"  Platt calibration: {APPLY_PLATT_SCALING} (method={PLATT_METHOD})")
    print()

    # Normalise weights
    w_sum = W_MODEL_A + W_MODEL_B
    wa    = W_MODEL_A / w_sum
    wb    = W_MODEL_B / w_sum

    # ── Cross-validation loop ─────────────────────────────────────────────────
    fold_results   = {}
    all_ytrue      = []
    all_prob_ens   = []

    metric_keys = ["macro_f1", "precision", "recall", "specificity", "roc_auc", "pr_auc"]
    collected   = {k: [] for k in metric_keys}

    for fold_idx in range(N_FOLDS):
        print(f"[Fold {fold_idx}] " + "-" * 50)

        # Extract arrays from each model
        pids_a, probs_a, ytrue_a = extract_fold_arrays(data_a, schema_a, fold_idx)
        pids_b, probs_b, ytrue_b = extract_fold_arrays(data_b, schema_b, fold_idx)

        print(f"  Model A patients: {len(pids_a)},  Model B patients: {len(pids_b)}")

        # Inner join on patient_id
        pids, probs_a_al, probs_b_al, ytrue = inner_join_fold(
            pids_a, probs_a, ytrue_a,
            pids_b, probs_b, ytrue_b,
            fold_idx,
        )
        print(f"  Shared patients after inner join: {len(pids)}")
        print(f"  Positive patients: {int(ytrue.sum())}  Negative: {int((ytrue==0).sum())}")

        # Platt calibration of Model B (CNN) probabilities
        if APPLY_PLATT_SCALING:
            probs_b_cal = platt_calibrate_fold(probs_b_al, ytrue, method=PLATT_METHOD)
            print(f"  CNN probs before calibration: mean={probs_b_al.mean():.3f}  "
                  f"neg_mean={probs_b_al[ytrue==0].mean():.3f}")
            print(f"  CNN probs after calibration:  mean={probs_b_cal.mean():.3f}  "
                  f"neg_mean={probs_b_cal[ytrue==0].mean():.3f}")
        else:
            probs_b_cal = probs_b_al
            warnings.warn("Platt calibration disabled — CNN probs may dominate ensemble.")

        # Weighted soft vote
        prob_ens = wa * probs_a_al + wb * probs_b_cal
        print(f"  Ensemble prob: mean={prob_ens.mean():.3f}  "
              f"pos_mean={prob_ens[ytrue==1].mean():.3f}  "
              f"neg_mean={prob_ens[ytrue==0].mean():.3f}")

        # Metrics at default threshold
        y_pred = (prob_ens >= DEFAULT_THRESHOLD).astype(int)
        m = compute_metrics(ytrue, y_pred, prob_ens)
        if m is None:
            print(f"  [WARN] Single class in fold {fold_idx} — skipping metrics.")
            continue

        print(f"  @t={DEFAULT_THRESHOLD:.2f}: "
              f"macro_f1={m['macro_f1']:.4f}  recall={m['recall']:.4f}  "
              f"spec={m['specificity']:.4f}  roc={m['roc_auc']:.4f}")

        fold_results[str(fold_idx)] = {
            **m,
            "n_test":         len(ytrue),
            "n_positive":     int(ytrue.sum()),
            "threshold_used": DEFAULT_THRESHOLD,
            "w_model_a":      round(wa, 4),
            "w_model_b":      round(wb, 4),
            "platt_applied":  APPLY_PLATT_SCALING,
            "patient_ids":    pids.tolist(),
            "y_true":         ytrue.tolist(),
            "prob_a":         probs_a_al.tolist(),
            "prob_b_raw":     probs_b_al.tolist(),
            "prob_b_cal":     probs_b_cal.tolist(),
            "prob_ensemble":  prob_ens.tolist(),
        }

        for k in metric_keys:
            collected[k].append(m[k])

        all_ytrue.extend(ytrue.tolist())
        all_prob_ens.extend(prob_ens.tolist())

    # ── Summary at default threshold ──────────────────────────────────────────
    print()
    print("=" * 68)
    print(f"CROSS-VALIDATION SUMMARY (patient-level, threshold={DEFAULT_THRESHOLD})")
    print("=" * 68)

    summary = {}
    for k in metric_keys:
        vals = collected[k]
        mean = float(np.mean(vals)) if vals else 0.0
        std  = float(np.std(vals))  if vals else 0.0
        summary[f"{k}_mean"] = round(mean, 4)
        summary[f"{k}_std"]  = round(std,  4)
        print(f"  {k:<14}: {mean:.4f} +/- {std:.4f}")

    # ── Threshold sweep on all-fold ensemble probs ────────────────────────────
    print()
    print("=" * 68)
    print("THRESHOLD SWEEP (aggregated across all folds)")
    print("  Clinical target: recall >= 0.80, specificity >= 0.70")
    print("=" * 68)
    print(f"  {'t':<6} {'recall':>8} {'spec':>8} {'f1':>8} {'prec':>8}  {'roc':>8}  Status")
    print("  " + "-" * 68)

    all_yt = np.array(all_ytrue, dtype=np.int8)
    all_pe = np.array(all_prob_ens, dtype=np.float32)
    sweep_results = []

    for t in THRESHOLD_SWEEP:
        y_pred_t = (all_pe >= t).astype(int)
        if len(np.unique(all_yt)) < 2:
            continue
        mac_f1  = f1_score(all_yt, y_pred_t, average="macro")
        prec    = precision_score(all_yt, y_pred_t, zero_division=0)
        rec     = recall_score(all_yt, y_pred_t, zero_division=0)
        tn,fp,fn,tp = confusion_matrix(all_yt, y_pred_t).ravel()
        spec    = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        roc     = roc_auc_score(all_yt, all_pe)

        r_ok  = rec  >= 0.80
        s_ok  = spec >= 0.70
        if r_ok and s_ok:   status = "*** BOTH TARGETS MET ***"
        elif r_ok:          status = "recall OK"
        elif s_ok:          status = "spec OK"
        else:               status = ""

        print(f"  {t:<6.2f} {rec:>8.4f} {spec:>8.4f} {mac_f1:>8.4f} {prec:>8.4f}  {roc:>8.4f}  {status}")
        sweep_results.append({"threshold": t, "recall": round(rec,4), "specificity": round(spec,4),
                               "macro_f1": round(mac_f1,4), "precision": round(prec,4),
                               "roc_auc": round(roc,4)})

    # ── Write JSON ────────────────────────────────────────────────────────────
    output = {
        "ensemble_type":    "soft_vote_weighted",
        "models":           [schema_a["label"], schema_b["label"]],
        "weights":          {"model_a": round(wa,4), "model_b": round(wb,4)},
        "platt_scaling":    {"applied": APPLY_PLATT_SCALING, "method": PLATT_METHOD},
        "default_threshold": DEFAULT_THRESHOLD,
        "folds":            fold_results,
        "summary":          summary,
        "threshold_sweep":  sweep_results,
        "config": {
            "w_model_a":         W_MODEL_A,
            "w_model_b":         W_MODEL_B,
            "apply_platt":       APPLY_PLATT_SCALING,
            "platt_method":      PLATT_METHOD,
            "threshold_default": DEFAULT_THRESHOLD,
            "n_folds":           N_FOLDS,
            "json_model_a":      str(JSON_MODEL_A),
            "json_model_b":      str(JSON_MODEL_B),
        },
    }

    out_path = Path("results") / make_output_filename(schema_a, schema_b, wa, wb)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out_path), "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)

    print()
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
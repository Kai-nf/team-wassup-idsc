from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from felicia.models import models as core


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "felicia" / "results" / "evaluation_metrics"
OUT_CSV = RESULT_DIR / "phase5_calibration_summary.csv"
OUT_PNG = RESULT_DIR / "phase5_calibration_plot.png"

EXPERIMENTS = [
    {"method": "method1", "model": "logistic", "resampler": "none"},
    {"method": "method1", "model": "rf", "resampler": "none"},
    {"method": "method2", "model": "logistic", "resampler": "none"},
    {"method": "method2", "model": "rf", "resampler": "none"},
    {"method": "method3_2", "model": "logistic", "resampler": "none"},
    {"method": "method3_2", "model": "rf", "resampler": "none"},
    {"method": "method4", "model": "logistic", "resampler": "smote"},
    {"method": "method4", "model": "logistic", "resampler": "borderline_smote"},
    {"method": "method4", "model": "logistic", "resampler": "adasyn"},
    {"method": "method4", "model": "rf", "resampler": "smote"},
    {"method": "method4", "model": "rf", "resampler": "borderline_smote"},
    {"method": "method4", "model": "rf", "resampler": "adasyn"},
]


def _experiment_label(config):
    if config["method"] == "method4":
        return f"{config['method']}|{config['model']}|{config['resampler']}"
    return f"{config['method']}|{config['model']}"


def _load_base(method: str):
    if method in {"method1", "method2"}:
        return core._load_method1_or_2_base("metadata.csv", "files")
    if method in {"method3", "method3_1", "method3_2"}:
        return core._load_method3_dataset(method, "metadata.csv")
    if method == "method4":
        return core._load_method4_dataset("Steve/dataset_v4_features_drop14.csv")
    raise ValueError(f"Unsupported method: {method}")


def _expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y_true)
    ece = 0.0
    for idx in range(bins):
        left = edges[idx]
        right = edges[idx + 1]
        if idx == bins - 1:
            mask = (y_prob >= left) & (y_prob <= right)
        else:
            mask = (y_prob >= left) & (y_prob < right)
        if not np.any(mask):
            continue
        acc = float(np.mean(y_true[mask]))
        conf = float(np.mean(y_prob[mask]))
        ece += abs(acc - conf) * (np.sum(mask) / total)
    return float(ece)


def _collect_out_of_fold_predictions(config):
    core.set_global_seed(core.GLOBAL_SEED)
    folds_data, _ = core.load_master_folds(core.ALLOWED_FOLD_FILE)
    normalized_folds = core.normalize_master_folds(folds_data)

    method = config["method"]
    model_name = config["model"]
    resampler = config["resampler"]

    if method in {"method1", "method2"}:
        X_base, y_base, pids_base = _load_base(method)
        patient_label_map = None
    elif method in {"method3", "method3_1", "method3_2"}:
        X_base, y_base, pids_base, patient_label_map = _load_base(method)
    else:
        X_base, y_base, pids_base, _ = _load_base(method)
        patient_label_map = None

    fold_rows = []
    all_true = []
    all_prob = []

    for fold_info in normalized_folds:
        train_pids = fold_info["train_patient_ids"]
        test_pids = fold_info["test_patient_ids"]

        train_mask = np.isin(pids_base, train_pids)
        test_mask = np.isin(pids_base, test_pids)

        X_train = X_base[train_mask]
        y_train = y_base[train_mask]
        pid_train = pids_base[train_mask]
        X_test = X_base[test_mask]
        y_test = y_base[test_mask]
        pid_test = pids_base[test_mask]

        if method == "method2":
            X_train = np.array([core.apply_clinical_filters(x) for x in X_train], dtype=np.float32)
            X_test = np.array([core.apply_clinical_filters(x) for x in X_test], dtype=np.float32)

        if X_train.ndim == 3:
            X_train, X_test, _ = core._scale_3d_train_test(X_train, X_test)
        else:
            scaler = core.StandardScaler()
            X_train = scaler.fit_transform(X_train).astype(np.float32)
            X_test = scaler.transform(X_test).astype(np.float32)

        if method in {"method1", "method2", "method3", "method3_1", "method3_2"}:
            X_train, y_train, pid_train = core.apply_domain_augmentation(X_train, y_train, pid_train, brugada_only=True)

        X_train_2d = core._flatten_if_needed(X_train)
        X_test_2d = core._flatten_if_needed(X_test)

        if method == "method4":
            X_train_2d, y_train = core.apply_feature_resampling(X_train_2d, y_train, strategy=resampler)

        best_model, _, _ = core._fit_with_nested_cv(model_name, X_train_2d, y_train)
        y_prob = best_model.predict_proba(X_test_2d)[:, 1]

        if method in {"method3", "method3_1", "method3_2"}:
            _, y_true_fold, _, y_prob_fold = core._aggregate_beats_to_patient(pid_test, y_test, y_prob)
        else:
            y_true_fold = y_test.astype(int)
            y_prob_fold = y_prob.astype(float)

        fold_rows.append(
            {
                "fold": int(fold_info["fold"]),
                "n_test": int(len(y_true_fold)),
                "n_positive": int(np.sum(y_true_fold == 1)),
                "brier_score": float(brier_score_loss(y_true_fold, y_prob_fold)),
                "ece": _expected_calibration_error(y_true_fold, y_prob_fold),
            }
        )
        all_true.append(y_true_fold)
        all_prob.append(y_prob_fold)

    return fold_rows, np.concatenate(all_true), np.concatenate(all_prob)


def main():
    rows = []
    plot_payload = []

    for config in EXPERIMENTS:
        fold_rows, y_true, y_prob = _collect_out_of_fold_predictions(config)
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
        label = _experiment_label(config)
        rows.append(
            {
                "label": label,
                "method": config["method"],
                "model": config["model"],
                "resampler": config["resampler"],
                "n_patients": int(len(y_true)),
                "n_positive": int(np.sum(y_true == 1)),
                "brier_score_mean": float(np.mean([r["brier_score"] for r in fold_rows])),
                "brier_score_std": float(np.std([r["brier_score"] for r in fold_rows])),
                "ece_mean": float(np.mean([r["ece"] for r in fold_rows])),
                "ece_std": float(np.std([r["ece"] for r in fold_rows])),
                "mean_predicted_probability": float(np.mean(y_prob)),
                "observed_prevalence": float(np.mean(y_true)),
                "fold_positive_counts": ",".join(str(r["n_positive"]) for r in fold_rows),
            }
        )
        plot_payload.append((label, prob_pred, prob_true))

    df = pd.DataFrame(rows).sort_values(["method", "model", "resampler"], na_position="last")
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved {OUT_CSV}")

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(4, 3, figsize=(14, 16), sharex=True, sharey=True)
        axes = axes.ravel()
        for ax, (label, prob_pred, prob_true) in zip(axes, plot_payload):
            ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
            ax.plot(prob_pred, prob_true, marker="o", linewidth=1.5)
            ax.set_title(label, fontsize=9)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.25)
        for ax in axes[-3:]:
            ax.set_xlabel("Predicted probability")
        for idx in [0, 3, 6, 9]:
            axes[idx].set_ylabel("Observed frequency")
        fig.suptitle("Phase 5 Calibration Curves from Out-of-Fold Predictions", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fig.savefig(OUT_PNG, dpi=150)
        plt.close(fig)
        print(f"Saved {OUT_PNG}")
    except Exception as exc:
        print(f"Calibration plot skipped: {exc}")


if __name__ == "__main__":
    main()

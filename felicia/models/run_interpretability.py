#!/usr/bin/env python3
# pyre-ignore-all-errors
"""
Phase 6: Interpretability & Explainable AI
Generates SHAP and Feature Importance plots for:
  - Random Forest (6.1 + 6.3)
  - Logistic Regression (6.1)

Train on FULL Method 4 feature dataset (all samples):
  → Used ONLY for generating interpretability plots.
  → Performance metrics still come from 5-fold CV results.

Outputs (saved to felicia/results/interpretability/):
  - rf_feature_importance.png        (6.3 RF)
  - lr_feature_importance.png        (6.3 LR - coefficient magnitudes)
  - rf_shap_beeswarm.png             (6.1 RF beeswarm)
  - lr_shap_beeswarm.png             (6.1 LR beeswarm)
  - rf_shap_waterfall_TP1.png        (6.1 RF waterfall for 1st Brugada patient)
  - rf_shap_waterfall_TP2.png        (6.1 RF waterfall for 2nd Brugada patient)
  - lr_shap_waterfall_TP1.png        (6.1 LR waterfall for 1st Brugada patient)
  - lr_shap_waterfall_TP2.png        (6.1 LR waterfall for 2nd Brugada patient)
"""

import argparse
from pathlib import Path
from typing import Any, List, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend, safe for scripts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────
# Styling constants for publication-quality plots
# ─────────────────────────────────────────────────────────
PALETTE_BRUGADA = "#E63946"   # Red for Brugada (positive class)
PALETTE_NORMAL  = "#457B9D"   # Blue for normal
DPI = 300
FONT_FAMILY = "DejaVu Sans"

plt.rcParams.update({
    "font.family":      FONT_FAMILY,
    "axes.titlesize":   14,
    "axes.labelsize":   12,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "figure.dpi":       DPI,
})

OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "interpretability"


def _load_data(
    features_csv: str = "Steve/dataset_v4_features_drop14.csv",
    metadata_csv: str = "metadata.csv",
) -> Tuple[Any, Any, List[str]]:
    """Load Method 4 features and binarize labels (Brugada=1, rest=0)."""
    df = pd.read_csv(features_csv)

    # The CSV contains a 'label' column (1 = Brugada, 0 = normal per Steve)
    if "label" in df.columns:
        y_arr: np.ndarray = df["label"].values.astype(int)
        feature_cols = [c for c in df.columns if c not in ("patient_id", "label")]
    else:
        # Fallback: join with metadata brugada column
        meta = pd.read_csv(metadata_csv)[["patient_id", "brugada"]]
        df = df.merge(meta, on="patient_id", how="left")
        # brugada: 1 = confirmed, 2 = suspected → treat both as positive
        y_arr = (df["brugada"] >= 1).astype(int).values
        feature_cols = [c for c in df.columns if c not in ("patient_id", "brugada")]

    X = df[feature_cols].astype(float)
    X = X.fillna(X.median())

    n_pos: int = int(y_arr.sum())
    n_neg: int = int((1 - y_arr).sum())
    print(f"[Data] {X.shape[0]} samples, {X.shape[1]} features")
    print(f"[Data] Class balance: {n_pos} Brugada / {n_neg} non-Brugada")

    return X, y_arr, feature_cols


def _train_rf(X_scaled: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    """Train RF on full dataset (interpretability only, not for metrics)."""
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_scaled, y)
    return rf


def _train_lr(X_scaled: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """Train LR on full dataset (interpretability only, not for metrics)."""
    lr = LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="saga",
        class_weight="balanced",
        max_iter=5000,
        random_state=42,
    )
    lr.fit(X_scaled, y)
    return lr


# ─────────────────────────────────────────────────────────
# 6.3 Feature Importance
# ─────────────────────────────────────────────────────────

def plot_rf_feature_importance(rf: RandomForestClassifier, feature_names: list, out_dir: Path):
    """Bar chart of top 10 features from RF feature_importances_."""
    importances = rf.feature_importances_
    top_n = min(10, len(feature_names))
    top_idx = np.argsort(importances)[-top_n:][::-1]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(
        range(top_n),
        importances[top_idx][::-1],
        color=PALETTE_BRUGADA,
        alpha=0.85,
        edgecolor="white",
    )
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in top_idx[::-1]], fontsize=10)
    ax.set_xlabel("Feature Importance (Gini)", fontsize=12)
    ax.set_title("Random Forest — Top 10 Feature Importances\n(Full Dataset, Method 4)", fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    plt.tight_layout()

    out_path = out_dir / "rf_feature_importance.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_lr_feature_importance(lr: LogisticRegression, feature_names: list, out_dir: Path):
    """Bar chart of top 10 features from LR coefficient magnitudes."""
    coefs = np.abs(lr.coef_[0])
    top_n = min(10, len(feature_names))
    top_idx = np.argsort(coefs)[-top_n:][::-1]

    # Use signed coefficients for colour coding (positive → pushes toward Brugada)
    signed = lr.coef_[0]
    colors = [PALETTE_BRUGADA if signed[i] > 0 else PALETTE_NORMAL for i in top_idx[::-1]]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(
        range(top_n),
        coefs[top_idx][::-1],
        color=colors,
        alpha=0.85,
        edgecolor="white",
    )
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in top_idx[::-1]], fontsize=10)
    ax.set_xlabel("|Coefficient| (Logistic Regression, L2)", fontsize=12)
    ax.set_title(
        "Logistic Regression — Top 10 Feature Coefficients\n"
        "(Full Dataset, Method 4 | Red=↑Brugada, Blue=↓Brugada)",
        fontsize=12,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    plt.tight_layout()

    out_path = out_dir / "lr_feature_importance.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────
# 6.1 SHAP Analysis
# ─────────────────────────────────────────────────────────

def _shap_values_positive_class(explainer, X_scaled: np.ndarray):
    """Return SHAP values for the positive (Brugada) class, compatible with shap 0.51+."""
    sv = explainer(X_scaled)
    # sv.values shape: (n_samples, n_features, n_classes) for multi-output, or (n_samples, n_features)
    if sv.values.ndim == 3:
        sv_pos = shap.Explanation(
            values=sv.values[:, :, 1],
            base_values=sv.base_values[:, 1] if sv.base_values.ndim == 2 else sv.base_values,
            data=sv.data,
            feature_names=sv.feature_names,
        )
    else:
        sv_pos = sv
    return sv_pos


def plot_shap_beeswarm(sv_pos, model_label: str, out_dir: Path):
    """SHAP beeswarm summary plot showing top-20 features."""
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.beeswarm(sv_pos, max_display=20, show=False)
    plt.title(f"{model_label} — SHAP Summary (Beeswarm)\n(Full Dataset, Method 4 Features)", fontsize=13)
    plt.tight_layout()
    out_path = out_dir / f"{model_label.lower().replace(' ', '_')}_shap_beeswarm.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_shap_waterfall(sv_pos, y: np.ndarray, patient_idx: int, model_label: str, tp_num: int, out_dir: Path):
    """SHAP waterfall plot for a single patient."""
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(sv_pos[patient_idx], max_display=15, show=False)
    label_str = "Brugada" if y[patient_idx] == 1 else "Non-Brugada"
    plt.title(
        f"{model_label} — SHAP Waterfall (Patient #{tp_num})\n"
        f"True Label: {label_str}",
        fontsize=13,
    )
    plt.tight_layout()
    slug = model_label.lower().replace(" ", "_")
    out_path = out_dir / f"{slug}_shap_waterfall_TP{tp_num}.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def run_rf_interpretability(X_scaled: np.ndarray, y: np.ndarray, feature_names: list, out_dir: Path):
    print("\n[RF] Training on full dataset…")
    rf = _train_rf(X_scaled, y)

    # 6.3 — Feature importance bar chart
    print("[RF] 6.3 Feature Importance…")
    plot_rf_feature_importance(rf, feature_names, out_dir)

    # 6.1 — SHAP
    print("[RF] 6.1 SHAP TreeExplainer…")
    explainer = shap.TreeExplainer(rf)
    sv_pos = _shap_values_positive_class(explainer, X_scaled)

    print("[RF] Beeswarm plot…")
    plot_shap_beeswarm(sv_pos, "Random Forest", out_dir)

    # Waterfall for 2 True Positive (Brugada) patients
    tp_indices = np.where(y == 1)[0]
    if len(tp_indices) >= 2:
        for num, idx in enumerate(tp_indices[:2], start=1):
            print(f"[RF] Waterfall TP{num} (patient index {idx})…")
            plot_shap_waterfall(sv_pos, y, int(idx), "Random Forest", num, out_dir)
    else:
        print("[RF] Warning: fewer than 2 Brugada patients found, skipping some waterfall plots.")

    print("[RF] Done.")


def run_lr_interpretability(X_scaled: np.ndarray, y: np.ndarray, feature_names: list, out_dir: Path):
    print("\n[LR] Training on full dataset…")
    lr = _train_lr(X_scaled, y)

    # 6.3 — Coefficient bar chart
    print("[LR] 6.3 Feature Importance (coefficients)…")
    plot_lr_feature_importance(lr, feature_names, out_dir)

    # 6.1 — SHAP LinearExplainer
    print("[LR] 6.1 SHAP LinearExplainer…")
    explainer = shap.LinearExplainer(lr, X_scaled, feature_names=feature_names)
    sv = explainer(X_scaled)

    print("[LR] Beeswarm plot…")
    plot_shap_beeswarm(sv, "Logistic Regression", out_dir)

    # Waterfall for 2 True Positive patients
    tp_indices = np.where(y == 1)[0]
    if len(tp_indices) >= 2:
        for num, idx in enumerate(tp_indices[:2], start=1):
            print(f"[LR] Waterfall TP{num} (patient index {idx})…")
            plot_shap_waterfall(sv, y, int(idx), "Logistic Regression", num, out_dir)
    else:
        print("[LR] Warning: fewer than 2 Brugada patients found.")

    print("[LR] Done.")


def main():
    parser = argparse.ArgumentParser(description="Phase 6 Interpretability — LR & RF SHAP + Feature Importance")
    parser.add_argument("--features-csv", default="Steve/dataset_v4_features_drop14.csv")
    parser.add_argument("--metadata-csv", default="metadata.csv")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # Load & scale data
    X_df, y, feature_names = _load_data(
        features_csv=getattr(args, "features_csv"),
        metadata_csv=getattr(args, "metadata_csv"),
    )
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)

    # Run RF
    run_rf_interpretability(X_scaled, y, feature_names, out_dir)

    # Run LR
    run_lr_interpretability(X_scaled, y, feature_names, out_dir)

    print(f"\n✅ All Phase 6 plots saved to: {out_dir}")


if __name__ == "__main__":
    main()

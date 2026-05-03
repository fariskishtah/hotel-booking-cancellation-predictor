"""
evaluate.py
===========
Generates evaluation plots and comparison tables for all trained models.
Saves figures to outputs/figures/ and prints a final summary table.
"""

import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, ConfusionMatrixDisplay,
)

sys.path.insert(0, str(Path(__file__).parent))

FIG_DIR   = Path(__file__).resolve().parent.parent / "outputs" / "figures"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
FIG_DIR.mkdir(parents=True, exist_ok=True)

BRIGHT_BLUE  = "#007BFF"
ACCENT_RED   = "#E63946"
PALETTE      = ["#007BFF","#E63946","#2DC653","#FFC300","#9B5DE5","#00B4D8","#FF6B6B"]

plt.rcParams["figure.dpi"] = 150
sns.set_theme(style="whitegrid")


def _save(fig, name: str):
    p = FIG_DIR / f"{name}.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  [eval] Saved → {p}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFUSION MATRIX
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_confusion_matrix(model, X_test, y_test, name="best_model"):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Not Cancelled", "Cancelled"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {name}", fontsize=13, fontweight="bold")
    _save(fig, f"11_confusion_matrix_{name.replace(' ','_')}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ROC CURVES (all models)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_roc_curves(models_dict: dict, X_test, y_test):
    """models_dict: {name: fitted_pipeline}"""
    fig, ax = plt.subplots(figsize=(8, 6))
    for (name, model), color in zip(models_dict.items(), PALETTE):
        proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})", color=color, lw=2)
    ax.plot([0,1],[0,1], "k--", lw=1.2)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Model Comparison", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    _save(fig, "12_roc_curves")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PRECISION-RECALL CURVES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_pr_curves(models_dict: dict, X_test, y_test):
    fig, ax = plt.subplots(figsize=(8, 6))
    for (name, model), color in zip(models_dict.items(), PALETTE):
        proba = model.predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, proba)
        ax.plot(rec, prec, label=name, color=color, lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    _save(fig, "13_precision_recall_curves")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODEL COMPARISON BAR CHART
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_model_comparison(results_df: pd.DataFrame):
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    df = results_df.set_index("model")[metrics]

    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 5))
    fig.suptitle("Model Comparison — Test Set Metrics", fontsize=14, fontweight="bold")

    for ax, metric in zip(axes, metrics):
        vals  = df[metric].sort_values(ascending=True)
        colors = [ACCENT_RED if v == vals.max() else BRIGHT_BLUE for v in vals]
        ax.barh(vals.index, vals.values, color=colors, edgecolor="white")
        ax.set_xlim(0.5, 1.0)
        ax.set_title(metric.upper().replace("_"," "), fontsize=10)
        for i, v in enumerate(vals.values):
            ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=8)

    _save(fig, "14_model_comparison")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FEATURE IMPORTANCE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_feature_importance(model, feature_names: list, top_n: int = 20):
    """Works for tree-based models (RF, XGB, LGBM) that have feature_importances_."""
    clf = model.named_steps["clf"]
    prep = model.named_steps["prep"]

    # Get all transformed feature names
    try:
        num_names = prep.transformers_[0][2]   # numeric feature names
        cat_names = prep.transformers_[1][2]   # categorical feature names
        all_names = list(num_names) + list(cat_names)
    except Exception:
        all_names = feature_names

    importances = clf.feature_importances_
    n = min(len(importances), len(all_names))
    imp_series = pd.Series(importances[:n], index=all_names[:n]).sort_values(ascending=False)
    imp_top = imp_series.head(top_n)

    fig, ax = plt.subplots(figsize=(9, 7))
    imp_top[::-1].plot(kind="barh", ax=ax, color=BRIGHT_BLUE, edgecolor="white")
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    _save(fig, "15_feature_importance")
    return imp_series


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_evaluation():
    from data_loader import load_raw
    from preprocessing import clean, engineer_features, prepare_data
    from train import get_base_models, run_training

    # Load saved artifacts if available, else retrain
    model_path   = MODEL_DIR / "best_model.pkl"
    results_path = MODEL_DIR / "model_results.csv"
    feats_path   = MODEL_DIR / "feature_names.pkl"

    raw = load_raw()
    df  = engineer_features(clean(raw))
    X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_data(df)

    if model_path.exists() and results_path.exists():
        best_model   = joblib.load(model_path)
        feature_names = joblib.load(feats_path) if feats_path.exists() else feature_names
        results_df   = pd.read_csv(results_path)
        print("[eval] Loaded saved model and results.")
    else:
        print("[eval] No saved model found — running training first …")
        best_model, results_df = run_training()

    # Train all base models for multi-model plots
    models_dict = {}
    for name, pipe in get_base_models(preprocessor).items():
        pipe.fit(X_train, y_train)
        models_dict[name] = pipe
    models_dict["Best (Tuned)"] = best_model

    print("\n[eval] Generating evaluation plots …")
    plot_confusion_matrix(best_model, X_test, y_test, "Best Model (Tuned)")
    plot_roc_curves(models_dict, X_test, y_test)
    plot_pr_curves(models_dict, X_test, y_test)
    plot_model_comparison(results_df)

    try:
        plot_feature_importance(best_model, feature_names)
    except Exception as e:
        print(f"  [eval] Feature importance skipped: {e}")

    print("\n[eval] ── Final Model Summary ────────────────────────────────")
    print(results_df[["model","accuracy","precision","recall","f1","roc_auc"]]
          .sort_values("roc_auc", ascending=False)
          .to_string(index=False))


if __name__ == "__main__":
    run_evaluation()

"""Model evaluation figures (ROC, PR, comparison charts, confusion matrices)."""
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve

from src import config

logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", context="notebook")

MODEL_COLORS = {
    "Logistic Regression": "#4C78A8", "Decision Tree": "#F58518", "Random Forest": "#54A24B",
    "XGBoost": "#B279A2", "Support Vector Machine (RBF)": "#72B7B2",
    "Artificial Neural Network (MLP)": "#FF9DA6", "Hybrid (RF + XGBoost)": "#E45756",
}


def _save(fig, name):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(predictions: pd.DataFrame, comparison: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 6))
    for key, name in config.MODEL_NAMES.items():
        fpr, tpr, _ = roc_curve(predictions["y_true"], predictions[f"prob_{key}"])
        auc = comparison.loc[comparison["Model"] == name, "ROC-AUC"].iloc[0]
        hybrid = key == config.HYBRID_KEY
        ax.plot(fpr, tpr, color=MODEL_COLORS[name], lw=3 if hybrid else 1.8, label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random guess")
    ax.set(xlabel="False positive rate (1 - specificity)", ylabel="True positive rate (sensitivity)",
           title="ROC curves on the test set")
    ax.legend(loc="lower right")
    _save(fig, "07_roc_curves.png")


def plot_pr_curves(predictions: pd.DataFrame, comparison: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 6))
    for key, name in config.MODEL_NAMES.items():
        prec, rec, _ = precision_recall_curve(predictions["y_true"], predictions[f"prob_{key}"])
        ap = comparison.loc[comparison["Model"] == name, "PR-AUC"].iloc[0]
        ax.plot(rec, prec, color=MODEL_COLORS[name], lw=3 if key == config.HYBRID_KEY else 1.8,
                label=f"{name} (AP = {ap:.3f})")
    prevalence = predictions["y_true"].mean()
    ax.axhline(prevalence, color="k", ls="--", lw=1, label=f"No-skill baseline ({prevalence:.2f})")
    ax.set(xlabel="Recall (sensitivity)", ylabel="Precision", title="Precision-Recall curves on the test set",
           ylim=(0, 1.02))
    ax.legend(loc="upper right")
    _save(fig, "08_precision_recall_curves.png")


def _grouped_bar(comparison, metrics, title, filename):
    """Grouped bars; the y-axis always starts at 0 so differences are not exaggerated."""
    data = comparison.melt(id_vars="Model", value_vars=metrics, var_name="Metric", value_name="Value")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.barplot(data=data, x="Metric", y="Value", hue="Model", palette=MODEL_COLORS, ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=6, padding=1)
    ax.set_ylim(0, 1.05)
    ax.set(title=title, ylabel="Score (axis starts at 0)", xlabel="")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=5, frameon=False)
    _save(fig, filename)


def plot_model_comparison(comparison: pd.DataFrame):
    _grouped_bar(comparison, ["ROC-AUC", "PR-AUC", "F1", "MCC"],
                 "Model comparison: ranking-quality and balanced metrics (test set)",
                 "09_model_comparison_auc_f1_mcc.png")
    _grouped_bar(comparison, ["Accuracy", "Precision", "Recall", "Specificity"],
                 "Model comparison: threshold-based metrics (test set)",
                 "10_model_comparison_accuracy_precision_recall_specificity.png")


def plot_confusion_matrix(counts: dict, model_name: str, filename: str):
    cm = np.array([[counts["TN"], counts["FP"]], [counts["FN"], counts["TP"]]])
    fig, ax = plt.subplots(figsize=(5, 4.3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Predicted: No churn", "Predicted: Churn"],
                yticklabels=["Actual: No churn", "Actual: Churn"], annot_kws={"size": 14})
    ax.set_title(f"Confusion matrix - {model_name}")
    _save(fig, filename)


def plot_all_confusion_matrices(confusion: dict):
    for key, name in config.MODEL_NAMES.items():
        fname = "11_confusion_matrix_hybrid.png" if key == config.HYBRID_KEY else f"11_confusion_matrix_{key}.png"
        plot_confusion_matrix(confusion[name], name, fname)


def plot_cv_boxplot(cv_folds: pd.DataFrame, metric="ROC-AUC"):
    """Per-fold distribution of a CV metric (shows the variability behind mean +/- std)."""
    data = cv_folds[cv_folds["Metric"] == metric]
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=data, x="Model", y="Value", hue="Model", palette=MODEL_COLORS, legend=False, ax=ax)
    sns.stripplot(data=data, x="Model", y="Value", color="black", size=4, ax=ax)
    ax.set(title=f"5-fold cross-validation: {metric} per fold (training data)", ylabel=metric, xlabel="")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    _save(fig, f"14_cv_{metric.lower().replace('-', '')}_boxplot.png")


def generate_all_model_plots(predictions, comparison, confusion, cv_folds=None):
    plot_roc_curves(predictions, comparison)
    plot_pr_curves(predictions, comparison)
    plot_model_comparison(comparison)
    plot_all_confusion_matrices(confusion)
    if cv_folds is not None:
        plot_cv_boxplot(cv_folds, "ROC-AUC")
    logger.info("Model figures saved to %s", config.FIGURES_DIR)

"""Static (matplotlib / seaborn) EDA figures saved to outputs/figures/."""
import logging

import matplotlib
matplotlib.use("Agg")  # headless backend: works without a display (and on Windows)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import config
from src.eda.exploratory_analysis import churn_rate_by
from src.features.feature_engineering import (ENGINEERED_BINARY, ENGINEERED_NUMERIC,
                                              ChurnFeatureEngineer, TENURE_LABELS)

logger = logging.getLogger(__name__)
PALETTE = {0: "#4C78A8", 1: "#E45756"}
LABELS = {0: "Not churned", 1: "Churned"}
sns.set_theme(style="whitegrid", context="notebook")


def _save(fig, name: str) -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _with_features(df: pd.DataFrame) -> pd.DataFrame:
    engineered = ChurnFeatureEngineer().transform(df[config.RAW_FEATURE_COLUMNS])
    engineered[config.TARGET_COLUMN] = df[config.TARGET_COLUMN].values
    return engineered


def plot_churn_distribution(df):
    counts = df[config.TARGET_COLUMN].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    bars = ax.bar([LABELS[i] for i in counts.index], counts.values,
                  color=[PALETTE[i] for i in counts.index])
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v}\n({100 * v / counts.sum():.1f}%)",
                ha="center", va="bottom")
    ax.set_ylim(0, counts.max() * 1.18)
    ax.set_title("Churn distribution (class imbalance)")
    ax.set_ylabel("Number of customers")
    _save(fig, "01_churn_distribution.png")


def plot_category_vs_churn(df, column, filename, title=None):
    """Churn rate (%) per category with the customer count printed on each bar."""
    rates = churn_rate_by(df, column).sort_values("churn_rate_pct", ascending=False)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar(rates[column].astype(str), rates["churn_rate_pct"], color="#E45756")
    for bar, n, r in zip(bars, rates["customers"], rates["churn_rate_pct"]):
        ax.text(bar.get_x() + bar.get_width() / 2, r, f"{r:.1f}%\n(n={n})", ha="center", va="bottom", fontsize=9)
    ax.axhline(100 * df[config.TARGET_COLUMN].mean(), ls="--", color="gray", label="Overall churn rate")
    ax.set_ylim(0, rates["churn_rate_pct"].max() * 1.25)
    ax.set_ylabel("Churn rate (%)")
    ax.set_title(title or f"Churn rate by {column}")
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    _save(fig, filename)


def plot_contract_vs_churn(df):
    plot_category_vs_churn(df, "Contract", "02_contract_vs_churn.png", "Churn rate by contract type")


def plot_tenure_vs_churn(df):
    fe = _with_features(df)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for val in (0, 1):
        axes[0].hist(fe.loc[fe[config.TARGET_COLUMN] == val, "tenure"], bins=36, alpha=0.65,
                     color=PALETTE[val], label=LABELS[val])
    axes[0].set(title="Tenure distribution by churn status", xlabel="Tenure (months)", ylabel="Customers")
    axes[0].legend()
    rates = churn_rate_by(fe, "tenure_group").set_index("tenure_group").reindex(TENURE_LABELS)
    axes[1].bar(rates.index, rates["churn_rate_pct"], color="#E45756")
    for i, r in enumerate(rates["churn_rate_pct"]):
        axes[1].text(i, r, f"{r:.1f}%", ha="center", va="bottom")
    axes[1].set(title="Churn rate by tenure group", xlabel="Tenure group (months)",
                ylabel="Churn rate (%)", ylim=(0, rates["churn_rate_pct"].max() * 1.2))
    _save(fig, "03_tenure_vs_churn.png")


def plot_monthly_charges_vs_churn(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.kdeplot(data=df, x="MonthlyCharges", hue=df[config.TARGET_COLUMN].map(LABELS), common_norm=False,
                fill=True, alpha=0.4, palette={LABELS[0]: PALETTE[0], LABELS[1]: PALETTE[1]}, ax=axes[0])
    axes[0].set(title="Monthly charges density by churn status", xlabel="Monthly charges")
    sns.boxplot(x=df[config.TARGET_COLUMN].map(LABELS), y=df["MonthlyCharges"],
                order=[LABELS[0], LABELS[1]], palette=[PALETTE[0], PALETTE[1]], hue=df[config.TARGET_COLUMN].map(LABELS),
                legend=False, ax=axes[1])
    axes[1].set(title="Monthly charges by churn status", xlabel="", ylabel="Monthly charges")
    _save(fig, "04_monthly_charges_vs_churn.png")


def plot_services_vs_churn(df):
    fe = _with_features(df)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    rates = churn_rate_by(fe, "service_count")
    axes[0].bar(rates["service_count"].astype(str), rates["churn_rate_pct"], color="#E45756")
    axes[0].set(title="Churn rate by number of subscribed services", xlabel="Service count",
                ylabel="Churn rate (%)")
    services = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    rows = []
    for s in services:
        sub = df[df["InternetService"] != "No"]
        for level in ("Yes", "No"):
            grp = sub[sub[s] == level]
            rows.append({"Service": s, "Subscribed": level, "churn_rate_pct": 100 * grp[config.TARGET_COLUMN].mean()})
    sns.barplot(data=pd.DataFrame(rows), x="Service", y="churn_rate_pct", hue="Subscribed",
                palette={"Yes": "#54A24B", "No": "#E45756"}, ax=axes[1])
    axes[1].set(title="Churn rate by add-on service (internet customers only)", ylabel="Churn rate (%)", xlabel="")
    plt.setp(axes[1].get_xticklabels(), rotation=25, ha="right")
    _save(fig, "05_services_vs_churn.png")


def plot_correlation_heatmap(df):
    fe = _with_features(df)
    cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"] + ENGINEERED_NUMERIC + ENGINEERED_BINARY
    corr = fe[cols + [config.TARGET_COLUMN]].corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(12, 9.5))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1,
                annot_kws={"size": 7}, cbar_kws={"label": "Pearson correlation"}, ax=ax)
    ax.set_title("Correlation heatmap (numeric and binary features, incl. churn)")
    _save(fig, "06_correlation_heatmap.png")


def generate_all_eda_plots(df):
    plot_churn_distribution(df)
    plot_contract_vs_churn(df)
    plot_tenure_vs_churn(df)
    plot_monthly_charges_vs_churn(df)
    plot_services_vs_churn(df)
    plot_correlation_heatmap(df)
    plot_category_vs_churn(df, "PaymentMethod", "06b_payment_method_vs_churn.png", "Churn rate by payment method")
    plot_category_vs_churn(df, "InternetService", "06c_internet_service_vs_churn.png", "Churn rate by internet service")
    logger.info("EDA figures saved to %s", config.FIGURES_DIR)

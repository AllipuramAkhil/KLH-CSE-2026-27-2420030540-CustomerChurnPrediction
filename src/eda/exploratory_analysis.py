"""Numeric EDA helpers (tables only; plots live in src/visualization/eda_plots.py).

EDA is purely descriptive: nothing is fitted and no modelling decision depends on
the test rows, so running it on the full cleaned dataset does not leak.
"""
import pandas as pd

from src import config


def dataset_overview(df: pd.DataFrame) -> dict:
    return {
        "n_rows": int(len(df)),
        "n_features": int(len(config.RAW_FEATURE_COLUMNS)),
        "churn_rate": float(df[config.TARGET_COLUMN].mean()),
        "non_churn_rate": float(1 - df[config.TARGET_COLUMN].mean()),
        "n_churn": int(df[config.TARGET_COLUMN].sum()),
        "n_non_churn": int((1 - df[config.TARGET_COLUMN]).sum()),
    }


def churn_rate_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Customers and churn rate (%) for each category of `column`."""
    grouped = df.groupby(column)[config.TARGET_COLUMN].agg(customers="size", churned="sum")
    grouped["churn_rate_pct"] = 100 * grouped["churned"] / grouped["customers"]
    return grouped.reset_index()


def iqr_outlier_report(df: pd.DataFrame, columns=("tenure", "MonthlyCharges", "TotalCharges")) -> pd.DataFrame:
    """Count of values outside 1.5 x IQR for each numeric column (reported, not removed)."""
    rows = []
    for col in columns:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        rows.append({"column": col, "lower_bound": lo, "upper_bound": hi,
                     "n_outliers": int(((df[col] < lo) | (df[col] > hi)).sum())})
    return pd.DataFrame(rows)


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    return df[["tenure", "MonthlyCharges", "TotalCharges"]].describe().T.reset_index(names="feature")

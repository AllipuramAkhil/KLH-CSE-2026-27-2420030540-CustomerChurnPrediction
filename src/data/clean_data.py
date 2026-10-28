"""Row-level data cleaning.

Everything in this module is *rule based*: no statistic (mean, median, category
frequency ...) is learned from the data here.  Statistics that must be learned
(imputation values, scaling, one-hot categories) live inside the sklearn
Pipeline (see preprocess.py) and are fitted on the training split only, which
is what prevents data leakage.
"""
import logging

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)

YES_NO_TO_INT = {"Yes": 1, "No": 0}


def _strip_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Remove leading/trailing whitespace from every text column and column name."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
            df[col] = df[col].astype("string").str.strip().astype(object)
    return df


def _numeric_or_nan(series: pd.Series) -> pd.Series:
    """Blank strings / junk become NaN, everything else becomes float."""
    if series.dtype == object:
        series = series.replace("", np.nan)
    return pd.to_numeric(series, errors="coerce")


def clean_raw_dataframe(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean the raw Telco dataframe and return (clean_df, data_quality_report)."""
    report: dict = {"rows_raw": int(len(raw)), "columns_raw": int(raw.shape[1])}
    df = _strip_strings(raw)

    # --- data types --------------------------------------------------------
    # TotalCharges is stored as text and contains blank strings for customers
    # with tenure == 0 (not billed yet).  Blank -> NaN; for tenure == 0 the true
    # value is logically 0 (a business rule, not a learned statistic).  Any other
    # NaN is left for the imputer inside the pipeline.
    df["TotalCharges"] = _numeric_or_nan(df["TotalCharges"])
    report["total_charges_blank_or_invalid"] = int(df["TotalCharges"].isna().sum())
    df["tenure"] = pd.to_numeric(df["tenure"], errors="raise").astype(int)
    new_customers = (df["tenure"] == 0) & df["TotalCharges"].isna()
    report["total_charges_set_to_zero_for_tenure_0"] = int(new_customers.sum())
    df.loc[new_customers, "TotalCharges"] = 0.0
    df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="raise").astype(int)
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="raise").astype(float)

    # --- duplicates --------------------------------------------------------
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)  # exact duplicate rows
    report["exact_duplicate_rows_removed"] = int(n_before - len(df))
    if config.ID_COLUMN in df.columns:
        report["duplicate_customer_ids"] = int(df[config.ID_COLUMN].duplicated().sum())
        # Rows with identical *features* but different IDs are legitimate
        # customers, so they are reported but kept.
        report["rows_with_identical_features_kept"] = int(
            df.duplicated(subset=config.RAW_FEATURE_COLUMNS).sum())

    # --- target encoding ---------------------------------------------------
    unknown = set(df[config.TARGET_COLUMN].dropna().unique()) - set(YES_NO_TO_INT)
    if unknown:
        raise ValueError(f"Unexpected values in target column '{config.TARGET_COLUMN}': {unknown}")
    df = df.dropna(subset=[config.TARGET_COLUMN]).copy()
    df[config.TARGET_COLUMN] = df[config.TARGET_COLUMN].map(YES_NO_TO_INT).astype(int)

    # --- constant columns (carry no information) ----------------------------
    constant_cols = [c for c in config.RAW_FEATURE_COLUMNS if df[c].nunique(dropna=False) <= 1]
    if constant_cols:
        logger.warning("Constant columns detected: %s", constant_cols)
    report["constant_columns"] = constant_cols

    # --- missing values report ----------------------------------------------
    missing = df[config.RAW_FEATURE_COLUMNS].isna().sum()
    report["missing_values_after_cleaning"] = {k: int(v) for k, v in missing.items() if v > 0}
    report["rows_clean"] = int(len(df))
    report["churn_rate"] = float(df[config.TARGET_COLUMN].mean())
    logger.info("Cleaning done: %d rows, churn rate %.2f%%", len(df), 100 * report["churn_rate"])
    return df.reset_index(drop=True), report


def split_features_target(df: pd.DataFrame):
    """Return (X, y, customer_ids).  The identifier is *never* a model feature."""
    X = df[config.RAW_FEATURE_COLUMNS].copy()
    y = df[config.TARGET_COLUMN].astype(int).copy()
    if config.ID_COLUMN in df.columns:
        ids = df[config.ID_COLUMN].copy()
    else:
        ids = pd.Series([f"row_{i}" for i in range(len(df))], index=df.index, name=config.ID_COLUMN)
    return X, y, ids


def prepare_features_for_inference(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same rule-based cleaning to *new* data (dashboard / batch use).

    Raises ValueError listing missing columns.  Extra columns are ignored.
    """
    df = _strip_strings(df)
    missing = [c for c in config.RAW_FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")
    X = df[config.RAW_FEATURE_COLUMNS].copy()
    if not pd.api.types.is_numeric_dtype(X["SeniorCitizen"]):
        X["SeniorCitizen"] = X["SeniorCitizen"].map({"Yes": 1, "No": 0, "1": 1, "0": 0})
    for col in ("SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"):
        X[col] = _numeric_or_nan(X[col])
    new_customers = (X["tenure"] == 0) & X["TotalCharges"].isna()
    X.loc[new_customers, "TotalCharges"] = 0.0
    return X

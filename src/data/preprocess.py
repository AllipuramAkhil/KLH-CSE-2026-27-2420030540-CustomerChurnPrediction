"""Leakage-safe preprocessing (sklearn ColumnTransformer).

The transformer is *unfitted* when created; it learns medians, means, standard
deviations and category lists only when `.fit()` is called on the TRAINING data
(inside the Pipeline, and inside every cross-validation fold).  The test set is
only ever passed through `.transform()`.
"""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.feature_engineering import (ENGINEERED_BINARY, ENGINEERED_CATEGORICAL,
                                              ENGINEERED_NUMERIC)

# Columns AFTER feature engineering
NUMERIC_FEATURES = (["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
                    + ENGINEERED_NUMERIC + ENGINEERED_BINARY)
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
] + ENGINEERED_CATEGORICAL


def build_preprocessor() -> ColumnTransformer:
    """Median-impute + standardise numerics; mode-impute + one-hot encode categoricals.

    Outliers: numeric charges/tenure are bounded business quantities without
    data-entry errors (see EDA IQR report), so no rows are removed.  Standard
    scaling is needed by Logistic Regression; tree models are unaffected by it,
    so ALL models share exactly the same preprocessing.
    """
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

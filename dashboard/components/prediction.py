"""Prediction helpers used by the single-customer form and the batch upload."""
import pandas as pd

from src import config
from src.data.clean_data import prepare_features_for_inference

CATEGORY_OPTIONS = {
    "gender": ["Female", "Male"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["No", "Yes", "No internet service"],
    "StreamingMovies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
}


def predict_frame(pipeline, raw_features: pd.DataFrame, threshold: float = config.DECISION_THRESHOLD):
    """Return a frame with churn_probability and churn_prediction (1 = churn)."""
    X = prepare_features_for_inference(raw_features)
    prob = pipeline.predict_proba(X)[:, 1]
    return pd.DataFrame({"churn_probability": prob, "churn_prediction": (prob >= threshold).astype(int)},
                        index=raw_features.index)


def find_id_column(df: pd.DataFrame):
    for col in df.columns:
        if col.strip().lower().replace("_", "") in ("customerid", "id"):
            return col
    return None


def unknown_categories(df: pd.DataFrame) -> dict:
    """Category values in an uploaded file that the model never saw (they are ignored by the encoder)."""
    problems = {}
    for col, options in CATEGORY_OPTIONS.items():
        if col in df.columns:
            bad = sorted(set(df[col].dropna().astype(str).str.strip()) - set(options))
            if bad:
                problems[col] = bad
    return problems

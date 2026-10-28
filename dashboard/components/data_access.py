"""Cached loaders: the dashboard only READS saved models/results; it never retrains."""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402
from src.data.clean_data import clean_raw_dataframe  # noqa: E402
from src.data.load_data import find_dataset, load_raw_data  # noqa: E402

NOT_RUN_MESSAGE = "Results not found. Run `python run.py` in the project folder first, then reload this page."


@st.cache_data(show_spinner=False)
def load_clean_data():
    """Cleaned dataset (for EDA pages) or None if the CSV is missing."""
    if find_dataset() is None:
        return None
    df, _ = clean_raw_dataframe(load_raw_data())
    return df


@st.cache_data(show_spinner=False)
def read_csv(name: str):
    path = config.METRICS_DIR / name
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def read_json(path_str: str):
    path = Path(path_str)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@st.cache_data(show_spinner=False)
def load_predictions():
    path = config.PREDICTIONS_DIR / "test_predictions.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_resource(show_spinner="Loading trained models ...")
def load_hybrid_pipeline():
    """Returns (pipeline, error_message).  Errors (missing file / version mismatch) are reported, not raised."""
    import joblib
    path = config.MODEL_PATHS[config.HYBRID_KEY]
    if not path.exists():
        return None, NOT_RUN_MESSAGE
    try:
        return joblib.load(path), None
    except Exception as exc:  # noqa: BLE001 - typically a scikit-learn/xgboost version mismatch
        return None, ("The saved model could not be loaded (" + str(exc)[:200] + "). "
                      "Re-create it in this environment with `python run.py`.")


def figure_path(name: str):
    path = config.FIGURES_DIR / name
    return path if path.exists() else None

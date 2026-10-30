"""Pipeline construction, stratified splitting, cross-validation and persistence."""
import logging

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from src import config
from src.data.preprocess import build_preprocessor
from src.evaluation.metrics import CV_SCORERS, compute_all_metrics
from src.features.feature_engineering import ChurnFeatureEngineer
from src.models.baseline_models import build_random_forest, build_xgboost, compute_scale_pos_weight, get_baseline_models
from src.models.hybrid_model import WeightedSoftVotingClassifier

logger = logging.getLogger(__name__)


def make_pipeline(estimator) -> Pipeline:
    """raw customer columns -> feature engineering -> preprocessing -> classifier."""
    return Pipeline([
        ("features", ChurnFeatureEngineer()),
        ("preprocess", build_preprocessor()),
        ("model", estimator),
    ])


def split_data(X, y, ids):
    """80/20 stratified split with a fixed seed.  The test part is kept untouched."""
    return train_test_split(X, y, ids, test_size=config.TEST_SIZE, stratify=y,
                            random_state=config.RANDOM_STATE)


def build_hybrid_estimator(y_train, weights=None, threshold=config.DECISION_THRESHOLD):
    weights = weights or config.HYBRID_WEIGHTS
    return WeightedSoftVotingClassifier(
        random_forest=build_random_forest(),
        xgboost=build_xgboost(compute_scale_pos_weight(y_train)),
        rf_weight=weights["random_forest"], xgb_weight=weights["xgboost"], threshold=threshold)


def build_all_pipelines(y_train, hybrid_weights=None) -> dict:
    """Unfitted pipelines for the 4 baselines + hybrid, keyed by model key."""
    pipelines = {key: make_pipeline(est) for key, est in get_baseline_models(y_train).items()}
    pipelines[config.HYBRID_KEY] = make_pipeline(build_hybrid_estimator(y_train, hybrid_weights))
    return pipelines


def cross_validate_models(pipelines: dict, X_train, y_train):
    """5-fold stratified CV on the TRAINING data only (the test set is never used).

    Returns (per_fold_df, summary_df).  Because preprocessing is inside the
    pipeline, imputer/scaler/encoder statistics are re-fitted on each fold's
    training part -> no leakage between CV folds either.
    """
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    fold_rows, summary_rows = [], []
    y_array = pd.Series(y_train).reset_index(drop=True)
    X_frame = X_train.reset_index(drop=True)
    for key, pipe in pipelines.items():
        name = config.MODEL_NAMES[key]
        logger.info("Cross-validating %s ...", name)
        fold_metric_values = {metric: [] for metric in CV_SCORERS}
        for fold_number, (fit_idx, valid_idx) in enumerate(cv.split(X_frame, y_array), start=1):
            X_fit, X_valid = X_frame.iloc[fit_idx], X_frame.iloc[valid_idx]
            y_fit, y_valid = y_array.iloc[fit_idx], y_array.iloc[valid_idx]
            fold_pipe = clone(pipe)

            # Derive XGBoost imbalance weight from this fold's training labels only.
            # Never use the labels of the CV validation fold to configure its model.
            if key == "xgboost":
                fold_pipe.set_params(model__scale_pos_weight=compute_scale_pos_weight(y_fit))
            elif key == config.HYBRID_KEY:
                fold_pipe.set_params(model__xgboost__scale_pos_weight=compute_scale_pos_weight(y_fit))

            fold_pipe.fit(X_fit, y_fit)
            probability = fold_pipe.predict_proba(X_valid)[:, 1]
            metrics = compute_all_metrics(y_valid, probability, threshold=config.DECISION_THRESHOLD)
            for metric, value in metrics.items():
                if metric in fold_metric_values:
                    fold_metric_values[metric].append(float(value))
                    fold_rows.append({"Model": name, "Fold": fold_number,
                                      "Metric": metric, "Value": float(value)})
        summary = {"Model": name}
        for metric, values in fold_metric_values.items():
            summary[f"{metric} Mean"] = float(pd.Series(values).mean())
            summary[f"{metric} Std"] = float(pd.Series(values).std(ddof=1))
        summary_rows.append(summary)
    return pd.DataFrame(fold_rows), pd.DataFrame(summary_rows)


def fit_pipelines(pipelines: dict, X_train, y_train) -> dict:
    """Fit every pipeline on the full training set."""
    fitted = {}
    for key, pipe in pipelines.items():
        logger.info("Fitting %s on %d training rows ...", config.MODEL_NAMES[key], len(X_train))
        fitted[key] = clone(pipe).fit(X_train, y_train)
    return fitted


def save_pipelines(fitted: dict) -> None:
    """Persist each fitted pipeline plus the fitted preprocessing steps (joblib)."""
    config.ensure_directories()
    for key, pipe in fitted.items():
        joblib.dump(pipe, config.MODEL_PATHS[key])
    # Feature engineering + ColumnTransformer only (identical for every model).
    joblib.dump(fitted[config.HYBRID_KEY][:-1], config.PREPROCESSOR_PATH)
    logger.info("Saved %d models to %s", len(fitted), config.MODELS_DIR)


def load_pipelines() -> dict:
    """Load all saved pipelines; raise a helpful error if training has not been run."""
    missing = [p for p in config.MODEL_PATHS.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Trained models not found. Run `python run.py` first.\nMissing: "
                                + ", ".join(str(p.name) for p in missing))
    return {key: joblib.load(path) for key, path in config.MODEL_PATHS.items()}

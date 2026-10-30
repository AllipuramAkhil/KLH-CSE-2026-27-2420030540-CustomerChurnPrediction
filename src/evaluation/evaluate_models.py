"""Held-out test-set evaluation and persistence of metrics / predictions."""
import json
import logging

import pandas as pd

from src import config
from src.evaluation.metrics import compute_all_metrics, confusion_counts

logger = logging.getLogger(__name__)


def evaluate_on_test(fitted: dict, X_test, y_test, ids_test, threshold=config.DECISION_THRESHOLD):
    """Evaluate every fitted pipeline on the untouched test set.

    Returns (comparison_df, predictions_df, confusion_dict).
    """
    rows, confusion = [], {}
    predictions = pd.DataFrame({"customer_id": ids_test.values, "y_true": y_test.values})
    for key, pipe in fitted.items():
        prob = pipe.predict_proba(X_test)[:, 1]
        pred = (prob >= threshold).astype(int)
        metrics = compute_all_metrics(y_test, prob, threshold)
        rows.append({"Model": config.MODEL_NAMES[key], **metrics})
        confusion[config.MODEL_NAMES[key]] = confusion_counts(y_test, pred)
        predictions[f"prob_{key}"] = prob
        predictions[f"pred_{key}"] = pred
    comparison = pd.DataFrame(rows)[["Model"] + config.METRIC_COLUMNS]
    return comparison, predictions, confusion


def save_evaluation(comparison, predictions, confusion, cv_folds, cv_summary) -> None:
    config.ensure_directories()
    comparison.to_csv(config.METRICS_DIR / "model_comparison.csv", index=False)
    comparison.to_json(config.METRICS_DIR / "model_comparison.json", orient="records", indent=2)
    predictions.to_csv(config.PREDICTIONS_DIR / "test_predictions.csv", index=False)
    with open(config.METRICS_DIR / "confusion_matrices.json", "w", encoding="utf-8") as fh:
        json.dump(confusion, fh, indent=2)
    if cv_folds is not None:
        cv_folds.to_csv(config.METRICS_DIR / "cross_validation_folds.csv", index=False)
    if cv_summary is not None:
        cv_summary.to_csv(config.METRICS_DIR / "cross_validation_summary.csv", index=False)
        cv_summary.to_json(config.METRICS_DIR / "cross_validation_summary.json",
                           orient="records", indent=2)
    logger.info("Saved metrics to %s", config.METRICS_DIR)

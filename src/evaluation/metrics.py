"""Classification metrics.  Positive class = Churn = 1.

Sensitivity (= Recall) = TP / (TP + FN): share of customers who really churned
that the model caught.  Specificity = TN / (TN + FP): share of customers who
really stayed that the model correctly left alone.
Every value is computed from real predictions with sklearn.
"""
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, cohen_kappa_score,
                             confusion_matrix, f1_score, log_loss, make_scorer,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)


def confusion_counts(y_true, y_pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}


def specificity_score(y_true, y_pred) -> float:
    c = confusion_counts(y_true, y_pred)
    denom = c["TN"] + c["FP"]
    return c["TN"] / denom if denom else 0.0


def sensitivity_score(y_true, y_pred) -> float:
    c = confusion_counts(y_true, y_pred)
    denom = c["TP"] + c["FN"]
    return c["TP"] / denom if denom else 0.0


def compute_all_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """All 11 requested metrics (+ confusion counts) from true labels and P(churn)."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Sensitivity": sensitivity_score(y_true, y_pred),
        "Specificity": specificity_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "PR-AUC": average_precision_score(y_true, y_prob),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Cohen's Kappa": cohen_kappa_score(y_true, y_pred),
        "Log Loss": log_loss(y_true, np.clip(y_prob, 1e-15, 1 - 1e-15), labels=[0, 1]),
    }


# Scorers for cross_validate (specificity / MCC / kappa are not built in as strings)
CV_SCORERS = {
    "Accuracy": "accuracy",
    "Precision": make_scorer(precision_score, zero_division=0),
    "Recall": "recall",
    "Specificity": make_scorer(specificity_score),
    "F1": "f1",
    "ROC-AUC": "roc_auc",
    "PR-AUC": "average_precision",
    "MCC": make_scorer(matthews_corrcoef),
    "Cohen's Kappa": make_scorer(cohen_kappa_score),
    "Log Loss": "neg_log_loss",  # sign flipped back to a positive loss afterwards
}

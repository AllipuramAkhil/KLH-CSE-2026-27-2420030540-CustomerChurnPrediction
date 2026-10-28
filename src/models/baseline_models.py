"""Baseline classifiers.

Hyper-parameters are fixed, sensible values (no heavy tuning): the goal is a
fair academic comparison, not leaderboard optimisation.  The Random Forest and
XGBoost defined here are also the two members of the hybrid, so the hybrid is
compared against *exactly* its own components.
"""
from collections import OrderedDict

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src import config


def build_random_forest(use_class_weight: bool = config.USE_CLASS_WEIGHTS) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=5, max_features="sqrt",
        class_weight="balanced_subsample" if use_class_weight else None,
        n_jobs=-1, random_state=config.RANDOM_STATE)


def build_xgboost(scale_pos_weight: float = 1.0) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=3, reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight, eval_metric="logloss", tree_method="hist",
        n_jobs=-1, random_state=config.RANDOM_STATE)


def compute_scale_pos_weight(y_train, use_class_weight: bool = config.USE_CLASS_WEIGHTS) -> float:
    """n_negative / n_positive of the TRAINING labels (1.0 if weighting is disabled)."""
    if not use_class_weight:
        return 1.0
    n_pos = float((y_train == 1).sum())
    return float((y_train == 0).sum()) / n_pos if n_pos else 1.0


def get_baseline_models(y_train, use_class_weight: bool = config.USE_CLASS_WEIGHTS) -> "OrderedDict":
    """Return {model_key: unfitted classifier} for the four baseline models."""
    class_weight = "balanced" if use_class_weight else None
    return OrderedDict([
        ("logistic_regression", LogisticRegression(
            C=1.0, max_iter=2000, class_weight=class_weight, solver="lbfgs",
            random_state=config.RANDOM_STATE)),
        ("decision_tree", DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=20, class_weight=class_weight,
            random_state=config.RANDOM_STATE)),
        ("random_forest", build_random_forest(use_class_weight)),
        ("xgboost", build_xgboost(compute_scale_pos_weight(y_train, use_class_weight))),
        ("svm", SVC(C=1.0, kernel="rbf", gamma="scale", probability=True,
                    class_weight=class_weight, random_state=config.RANDOM_STATE)),
        ("ann", MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                              solver="adam", alpha=0.001, learning_rate_init=0.001,
                              max_iter=350, early_stopping=True, validation_fraction=0.1,
                              n_iter_no_change=20, random_state=config.RANDOM_STATE)),
    ])

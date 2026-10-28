"""Proposed model: Random Forest + XGBoost weighted soft-voting ensemble.

    P_hybrid(churn) = w_rf * P_RF(churn) + w_xgb * P_XGB(churn)

with w_rf + w_xgb = 1 (weights are normalised).  Both members are trained on
the same (already preprocessed) training data.  The class prediction is
`P_hybrid >= threshold`.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone


class WeightedSoftVotingClassifier(ClassifierMixin, BaseEstimator):
    """Binary soft-voting ensemble of a Random Forest and an XGBoost classifier."""

    def __init__(self, random_forest=None, xgboost=None, rf_weight=0.5, xgb_weight=0.5,
                 threshold=0.5):
        self.random_forest = random_forest
        self.xgboost = xgboost
        self.rf_weight = rf_weight
        self.xgb_weight = xgb_weight
        self.threshold = threshold

    # -- helpers -----------------------------------------------------------
    def _normalised_weights(self):
        if self.rf_weight < 0 or self.xgb_weight < 0:
            raise ValueError("Ensemble weights must be non-negative.")
        total = self.rf_weight + self.xgb_weight
        if total <= 0:
            raise ValueError("At least one ensemble weight must be positive.")
        return self.rf_weight / total, self.xgb_weight / total

    # -- sklearn API -------------------------------------------------------
    def fit(self, X, y):
        if self.random_forest is None or self.xgboost is None:
            raise ValueError("Both `random_forest` and `xgboost` estimators must be provided.")
        self._normalised_weights()  # validate early
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("WeightedSoftVotingClassifier supports binary targets only.")
        self.rf_ = clone(self.random_forest).fit(X, y)
        self.xgb_ = clone(self.xgboost).fit(X, y)
        return self

    def component_probabilities(self, X):
        """Churn probability of each member: returns (P_RF, P_XGB)."""
        return self.rf_.predict_proba(X)[:, 1], self.xgb_.predict_proba(X)[:, 1]

    def predict_proba(self, X):
        w_rf, w_xgb = self._normalised_weights()
        p_rf, p_xgb = self.component_probabilities(X)
        p_hybrid = w_rf * p_rf + w_xgb * p_xgb
        return np.column_stack([1.0 - p_hybrid, p_hybrid])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)

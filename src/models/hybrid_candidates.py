"""Research hybrid candidates for the customer churn project.

All estimators consume the same fold-fitted feature pipeline. Candidate stacking
uses sklearn's internal cross-validated predictions for its meta learner.
"""
from sklearn.base import clone
from sklearn.ensemble import StackingClassifier, VotingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src import config
from src.models.baseline_models import build_random_forest, build_xgboost


def build_base_estimators(y_train, use_fold_local_xgb_weight=True):
    n_pos = max(1, int((y_train == 1).sum()))
    n_neg = int((y_train == 0).sum())
    # Stacking has an internal CV layer. Use an unweighted XGBoost member there
    # so its weight does not depend on labels from the internal validation fold.
    xgb_weight = (n_neg / n_pos) if use_fold_local_xgb_weight else 1.0
    return [
        ("rf", build_random_forest()),
        ("xgb", build_xgboost(xgb_weight)),
        ("svm", SVC(C=1.0, kernel="rbf", probability=True,
                    class_weight="balanced", random_state=config.RANDOM_STATE)),
    ]


def build_stacking_candidate(y_train):
    """RF + XGB + SVM, with Logistic Regression meta learner."""
    return StackingClassifier(
        estimators=build_base_estimators(y_train, use_fold_local_xgb_weight=False),
        final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced",
                                           random_state=config.RANDOM_STATE),
        stack_method="predict_proba", cv=3, n_jobs=1, passthrough=False
    )


def build_weighted_voting_candidate(y_train, weights=(0.2, 0.6, 0.2)):
    """Weighted soft voting with a diversified RF/XGB/SVM set."""
    estimators = build_base_estimators(y_train)
    return VotingClassifier(estimators=estimators, voting="soft",
                            weights=list(weights), n_jobs=1)


def build_rf_select_ann_candidate(y_train):
    """RF importance selection followed by a small ANN."""
    selector = SelectFromModel(build_random_forest(), threshold="median")
    ann = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300,
                        early_stopping=True, validation_fraction=0.1,
                        n_iter_no_change=15, alpha=0.001, random_state=config.RANDOM_STATE)
    return Pipeline([("select", selector), ("ann", ann)])


def build_leaf_embedding_ann_candidate(y_train):
    """XGBoost leaf-index representation followed by an ANN.

    XGBoost's apply() is used to generate leaf indices. The transformer is
    implemented as a sklearn-compatible estimator to remain fold-local.
    """
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.preprocessing import OneHotEncoder

    class LeafIndexEncoder(BaseEstimator, TransformerMixin):
        def __init__(self, n_estimators=120, max_depth=3, random_state=42):
            self.n_estimators = n_estimators
            self.max_depth = max_depth
            self.random_state = random_state

        def fit(self, X, y):
            self.model_ = XGBClassifier(
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss", tree_method="hist", n_jobs=1,
                random_state=self.random_state,
                scale_pos_weight=max(1, int((y == 0).sum())) / max(1, int((y == 1).sum()))
            )
            self.model_.fit(X, y)
            leaves = self.model_.apply(X)
            self.encoder_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            self.encoder_.fit(leaves)
            return self

        def transform(self, X):
            return self.encoder_.transform(self.model_.apply(X))

    return Pipeline([
        ("leaf_encoder", LeafIndexEncoder(random_state=config.RANDOM_STATE)),
        ("ann", MLPClassifier(hidden_layer_sizes=(64,), max_iter=250,
                              early_stopping=True, random_state=config.RANDOM_STATE))
    ])


def build_stat_select_voting_candidate(y_train):
    """Chi-square feature selection followed by soft voting.

    MinMaxScaler makes the transformed inputs non-negative for chi-square.
    This candidate selects 70% of encoded features (at least one) inside each
    training fold; the selector never sees validation labels.
    """
    estimators = [
        ("rf", build_random_forest()),
        ("xgb", build_xgboost(max(1, int((y_train == 0).sum())) /
                              max(1, int((y_train == 1).sum())))),
    ]
    return Pipeline([
        ("minmax", MinMaxScaler()),
        ("select", SelectKBest(score_func=chi2, k=25)),
        ("voting", VotingClassifier(estimators=estimators, voting="soft", weights=[0.4, 0.6], n_jobs=1))
    ])


def candidate_factories(y_train):
    """Return named estimators to be wrapped by the common preprocessing pipeline."""
    return {
        "stacking_rf_xgb_svm_lr": build_stacking_candidate(y_train),
        "weighted_soft_voting_rf_xgb_svm": build_weighted_voting_candidate(y_train),
        "rf_feature_selection_ann": build_rf_select_ann_candidate(y_train),
        "xgb_leaf_embedding_ann": build_leaf_embedding_ann_candidate(y_train),
        "chi2_feature_selection_ensemble": build_stat_select_voting_candidate(y_train),
    }

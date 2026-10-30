"""Training-only RandomizedSearchCV for leading hybrid candidates.

The final test set is not passed to the search. All selection is done inside
stratified five-fold CV on the training partition. Run after Stage 4 candidate
experiments with:
    python -m src.evaluation.tune_hybrid_candidates
"""
import json
import logging
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from scipy.stats import randint, loguniform

from src import config
from src.data.clean_data import clean_raw_dataframe, split_features_target
from src.data.load_data import ensure_dataset, load_raw_data
from src.models.model_training import split_data, make_pipeline
from src.models.hybrid_candidates import candidate_factories

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def build_searches(y_train):
    # Stacking and voting have the same base learners; these search spaces are
    # intentionally modest to keep runtime practical and results reproducible.
    factories = candidate_factories(y_train)
    stack = make_pipeline(factories["stacking_rf_xgb_svm_lr"])
    vote = make_pipeline(factories["weighted_soft_voting_rf_xgb_svm"])
    # Keep the XGBoost imbalance weight fixed during CV. Deriving it from all
    # outer-training labels before RandomizedSearchCV would let each inner
    # validation fold contribute to this hyperparameter. This is deliberately
    # conservative; the CV procedure now selects model parameters without
    # using validation-fold prevalence to set scale_pos_weight.
    stack.set_params(model__xgb__scale_pos_weight=1.0)
    vote.set_params(model__xgb__scale_pos_weight=1.0)
    stack_space = {
        "model__final_estimator__C": loguniform(0.03, 30),
        "model__rf__n_estimators": randint(100, 301),
        "model__rf__max_depth": [None, 8, 12, 18],
        "model__xgb__n_estimators": randint(100, 301),
        "model__xgb__max_depth": randint(2, 6),
        "model__xgb__learning_rate": loguniform(0.02, 0.15),
        "model__svm__C": loguniform(0.1, 10),
    }
    vote_space = {
        "model__weights": [[0.2, 0.6, 0.2], [0.1, 0.7, 0.2],
                           [0.2, 0.7, 0.1], [0.3, 0.5, 0.2],
                           [0.25, 0.5, 0.25], [0.15, 0.6, 0.25]],
        "model__rf__n_estimators": randint(100, 301),
        "model__rf__max_depth": [None, 8, 12, 18],
        "model__xgb__n_estimators": randint(100, 301),
        "model__xgb__max_depth": randint(2, 6),
        "model__xgb__learning_rate": loguniform(0.02, 0.15),
        "model__svm__C": loguniform(0.1, 10),
    }
    return {
        "stacking_rf_xgb_svm_lr": (stack, stack_space),
        "weighted_soft_voting_rf_xgb_svm": (vote, vote_space),
    }


def main():
    start = time.time()
    raw = load_raw_data(ensure_dataset())
    clean, _ = clean_raw_dataframe(raw)
    X, y, ids = split_features_target(clean)
    X_train, X_test, y_train, y_test, ids_train, ids_test = split_data(X, y, ids)
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    searches = build_searches(y_train)
    summaries, cv_tables = [], []
    for name, (pipe, space) in searches.items():
        logging.info("Tuning %s", name)
        search = RandomizedSearchCV(
            estimator=pipe, param_distributions=space, n_iter=8,
            scoring={"roc_auc": "roc_auc", "average_precision": "average_precision",
                     "f1": "f1"},
            refit="average_precision", cv=cv, n_jobs=1, random_state=config.RANDOM_STATE,
            verbose=1, return_train_score=False, error_score="raise"
        )
        search.fit(X_train, y_train)
        cv_result = pd.DataFrame(search.cv_results_)
        cv_result.insert(0, "Candidate", name)
        cv_tables.append(cv_result)
        best = search.best_estimator_
        # Test is touched only after the CV search has selected its best config.
        prob = best.predict_proba(X_test)[:, 1]
        pred = (prob >= config.DECISION_THRESHOLD).astype(int)
        summaries.append({
            "Candidate": name, "Best CV PR-AUC": search.best_score_,
            "Best CV ROC-AUC": search.cv_results_["mean_test_roc_auc"][search.best_index_],
            "Best CV F1": search.cv_results_["mean_test_f1"][search.best_index_],
            "Test ROC-AUC": roc_auc_score(y_test, prob),
            "Test PR-AUC": average_precision_score(y_test, prob),
            "Test F1": f1_score(y_test, pred),
            "Best parameters": json.dumps(search.best_params_, default=str),
        })
        model_path = config.HYBRID_MODEL_DIR / f"{name}_tuned.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        import joblib
        joblib.dump(best, model_path)
        logging.info("Best %s CV PR-AUC %.4f; test PR-AUC %.4f",
                     name, search.best_score_, average_precision_score(y_test, prob))

    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(config.METRICS_DIR / "hybrid_tuning_summary.csv", index=False)
    pd.concat(cv_tables, ignore_index=True).to_csv(config.METRICS_DIR / "hybrid_tuning_cv_results.csv", index=False)
    meta = {"completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_iter_per_candidate": 8, "cv_folds": 5,
            "selection_metric": "average_precision (PR-AUC)",
            "random_state": config.RANDOM_STATE, "runtime_seconds": round(time.time()-start, 2),
            "test_set_used_only_after_cv_selection": True}
    (config.REPORTS_DIR / "hybrid_tuning_metadata.json").write_text(json.dumps(meta, indent=2))
    print(pd.DataFrame(summaries)[["Candidate","Best CV PR-AUC","Best CV ROC-AUC","Best CV F1",
                                    "Test ROC-AUC","Test PR-AUC","Test F1"]].round(4).to_string(index=False))
    print("Saved tuned results in", config.METRICS_DIR)


if __name__ == "__main__":
    main()

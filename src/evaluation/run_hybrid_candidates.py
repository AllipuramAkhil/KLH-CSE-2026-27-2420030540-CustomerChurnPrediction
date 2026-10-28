"""Run comparative hybrid-candidate experiments.

Usage:
    python -m src.evaluation.run_hybrid_candidates

Uses the prescribed dataset and the same fixed stratified split as the main
pipeline. Candidate CV is five-fold, training-only. Each candidate is freshly
constructed from each fold's training labels to avoid fold-level weighting
leakage. Results are saved to outputs/metrics and outputs/predictions.
"""
import argparse
import json
import logging
import time
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src import config
from src.data.clean_data import clean_raw_dataframe, split_features_target
from src.data.load_data import ensure_dataset, load_raw_data
from src.evaluation.metrics import compute_all_metrics
from src.models.model_training import split_data, make_pipeline
from src.models.hybrid_candidates import candidate_factories

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def fit_candidate(name, y_fit):
    estimator = candidate_factories(y_fit)[name]
    return make_pipeline(estimator)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None, help="optional exact candidate key to run alone")
    args = parser.parse_args()
    start = time.time()
    raw = load_raw_data(ensure_dataset())
    clean, _ = clean_raw_dataframe(raw)
    X, y, ids = split_features_target(clean)
    X_train, X_test, y_train, y_test, ids_train, ids_test = split_data(X, y, ids)
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    names = list(candidate_factories(y_train).keys())
    if args.only:
        if args.only not in names:
            raise ValueError(f"Unknown candidate {args.only!r}. Choose from {names}")
        names = [args.only]
    folds = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True,
                            random_state=config.RANDOM_STATE)
    cv_rows, test_rows, pred_data = [], [], {"customer_id": ids_test.reset_index(drop=True),
                                             "y_true": y_test}
    for name in names:
        logging.info("Candidate: %s", name)
        fold_auc, fold_pr, fold_f1 = [], [], []
        for fold_i, (fit_idx, valid_idx) in enumerate(folds.split(X_train, y_train), start=1):
            X_fit, X_valid = X_train.iloc[fit_idx], X_train.iloc[valid_idx]
            y_fit, y_valid = y_train.iloc[fit_idx], y_train.iloc[valid_idx]
            pipe = fit_candidate(name, y_fit)
            pipe.fit(X_fit, y_fit)
            prob = pipe.predict_proba(X_valid)[:, 1]
            metrics = compute_all_metrics(y_valid, prob, config.DECISION_THRESHOLD)
            fold_auc.append(metrics["ROC-AUC"]); fold_pr.append(metrics["PR-AUC"]); fold_f1.append(metrics["F1"])
            cv_rows.append({"Candidate": name, "Fold": fold_i, **metrics})
            logging.info("  fold %d/%d ROC-AUC=%.4f PR-AUC=%.4f F1=%.4f",
                         fold_i, config.CV_FOLDS, metrics["ROC-AUC"], metrics["PR-AUC"], metrics["F1"])
        final_pipe = fit_candidate(name, y_train)
        final_pipe.fit(X_train, y_train)
        prob = final_pipe.predict_proba(X_test)[:, 1]
        metrics = compute_all_metrics(y_test, prob, config.DECISION_THRESHOLD)
        test_rows.append({"Candidate": name, **metrics,
                          "CV ROC-AUC Mean": float(np.mean(fold_auc)),
                          "CV ROC-AUC SD": float(np.std(fold_auc, ddof=1)),
                          "CV PR-AUC Mean": float(np.mean(fold_pr)),
                          "CV F1 Mean": float(np.mean(fold_f1))})
        pred_data[f"prob_{name}"] = prob
        pred_data[f"pred_{name}"] = (prob >= config.DECISION_THRESHOLD).astype(int)
        logging.info("  holdout ROC-AUC=%.4f PR-AUC=%.4f F1=%.4f",
                     metrics["ROC-AUC"], metrics["PR-AUC"], metrics["F1"])

    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    config.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    fold_path = config.METRICS_DIR / "hybrid_candidate_cv_folds.csv"
    summary_path = config.METRICS_DIR / "hybrid_candidate_comparison.csv"
    pred_path = config.PREDICTIONS_DIR / "hybrid_candidate_test_predictions.csv"
    new_folds = pd.DataFrame(cv_rows)
    summary = pd.DataFrame(test_rows)
    new_predictions = pd.DataFrame(pred_data)
    if args.only and fold_path.exists() and summary_path.exists() and pred_path.exists():
        old_folds = pd.read_csv(fold_path)
        old_summary = pd.read_csv(summary_path)
        old_predictions = pd.read_csv(pred_path)
        old_folds = old_folds[~old_folds["Candidate"].isin(names)]
        old_summary = old_summary[~old_summary["Candidate"].isin(names)]
        old_prediction_cols = [c for c in old_predictions.columns
                               if not any(c.endswith(n) or c in {f"prob_{n}", f"pred_{n}"} for n in names)]
        old_predictions = old_predictions[old_prediction_cols]
        new_folds = pd.concat([old_folds, new_folds], ignore_index=True)
        summary = pd.concat([old_summary, summary], ignore_index=True)
        for col in new_predictions.columns:
            if col not in {"customer_id", "y_true"}:
                old_predictions = old_predictions.drop(columns=[col], errors="ignore")
                old_predictions[col] = new_predictions[col]
        new_predictions = old_predictions
    new_folds.to_csv(fold_path, index=False)
    summary = summary.sort_values("CV ROC-AUC Mean", ascending=False)
    summary.to_csv(summary_path, index=False)
    new_predictions.to_csv(pred_path, index=False)
    metadata = {"finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "dataset": config.RAW_DATA_FILENAME, "n_train": len(X_train), "n_test": len(X_test),
                "cv_folds": config.CV_FOLDS, "random_state": config.RANDOM_STATE,
                "candidates": names, "runtime_seconds": round(time.time()-start, 2)}
    (config.REPORTS_DIR / "hybrid_candidate_run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print("\nCandidate comparison:")
    print(summary[["Candidate","CV ROC-AUC Mean","CV ROC-AUC SD","CV PR-AUC Mean","CV F1 Mean",
                   "Accuracy","F1","ROC-AUC","PR-AUC","MCC","Log Loss"]].round(4).to_string(index=False))
    print(f"\nSaved candidate results under {config.METRICS_DIR}")


if __name__ == "__main__":
    main()

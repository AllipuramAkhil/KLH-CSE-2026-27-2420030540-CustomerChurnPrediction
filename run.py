"""End-to-end pipeline:  python run.py

1 check dataset -> 2 load -> 3 clean -> 4 split -> 5 EDA -> 6 cross-validate ->
7 train -> 8 evaluate -> 9 save models/metrics -> 10 plots, importance, SHAP.

Options:  --skip-cv   --skip-shap   --rf-weight 0.5 --xgb-weight 0.5
"""
import argparse
import json
import logging
import platform
import sys
import time
import warnings
from datetime import datetime
from importlib import metadata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="shap")

from src import config  # noqa: E402
from src.data.clean_data import clean_raw_dataframe, split_features_target  # noqa: E402
from src.data.load_data import ensure_dataset, load_raw_data  # noqa: E402
from src.eda.exploratory_analysis import dataset_overview, iqr_outlier_report, numeric_summary  # noqa: E402
from src.evaluation import model_comparison as mc  # noqa: E402
from src.evaluation.evaluate_models import evaluate_on_test, save_evaluation  # noqa: E402
from src.explainability.shap_analysis import run_shap_analysis  # noqa: E402
from src.models import model_training as mt  # noqa: E402
from src.visualization import eda_plots, feature_importance, model_plots  # noqa: E402

logger = logging.getLogger("run")
PACKAGES = ["pandas", "numpy", "scikit-learn", "xgboost", "matplotlib", "seaborn", "plotly",
            "streamlit", "joblib", "shap", "scipy"]


def step(number: int, text: str) -> None:
    logger.info("[%d/10] %s", number, text)


def package_versions() -> dict:
    versions = {}
    for pkg in PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = "not installed"
    return versions


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the customer churn prediction pipeline.")
    parser.add_argument("--skip-cv", action="store_true", help="skip 5-fold cross-validation")
    parser.add_argument("--skip-shap", action="store_true", help="skip the (slowest) SHAP step")
    parser.add_argument("--rf-weight", type=float, default=config.HYBRID_WEIGHTS["random_forest"])
    parser.add_argument("--xgb-weight", type=float, default=config.HYBRID_WEIGHTS["xgboost"])
    args = parser.parse_args()
    weights = {"random_forest": args.rf_weight, "xgboost": args.xgb_weight}

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    start = time.time()
    config.ensure_directories()

    step(1, "Checking dataset")
    dataset_path = ensure_dataset()
    logger.info("Using dataset: %s", dataset_path)

    step(2, "Loading data")
    raw = load_raw_data(dataset_path)

    step(3, "Cleaning data")
    df, quality = clean_raw_dataframe(raw)
    (config.REPORTS_DIR / "data_quality_report.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    df.to_csv(config.PROCESSED_DATA_DIR / "telco_clean.csv", index=False)

    step(4, "Stratified 80/20 train/test split (test set stays untouched until step 8)")
    X, y, ids = split_features_target(df)
    X_train, X_test, y_train, y_test, ids_train, ids_test = mt.split_data(X, y, ids)
    logger.info("Train: %d rows (churn %.2f%%) | Test: %d rows (churn %.2f%%)",
                len(X_train), 100 * y_train.mean(), len(X_test), 100 * y_test.mean())
    X_train.assign(**{config.TARGET_COLUMN: y_train}).to_csv(config.PROCESSED_DATA_DIR / "train.csv", index=False)
    X_test.assign(**{config.TARGET_COLUMN: y_test}).to_csv(config.PROCESSED_DATA_DIR / "test.csv", index=False)

    step(5, "Exploratory data analysis (descriptive only)")
    (config.REPORTS_DIR / "eda_summary.json").write_text(json.dumps(dataset_overview(df), indent=2), encoding="utf-8")
    iqr_outlier_report(df).to_csv(config.METRICS_DIR / "outlier_report_iqr.csv", index=False)
    numeric_summary(df).to_csv(config.METRICS_DIR / "numeric_summary.csv", index=False)
    eda_plots.generate_all_eda_plots(df)

    pipelines = mt.build_all_pipelines(y_train, weights)

    step(6, "Cross-validation (training data only)")
    cv_folds = cv_summary = None
    if args.skip_cv:
        logger.info("Skipped (--skip-cv)")
    else:
        cv_folds, cv_summary = mt.cross_validate_models(pipelines, X_train, y_train)

    step(7, "Training final models on the full training set")
    fitted = mt.fit_pipelines(pipelines, X_train, y_train)

    step(8, "Evaluating on the held-out test set")
    comparison, predictions, confusion = evaluate_on_test(fitted, X_test, y_test, ids_test)
    print("\n" + comparison.round(4).to_string(index=False) + "\n")

    step(9, "Saving models, metrics and predictions")
    mt.save_pipelines(fitted)
    save_evaluation(comparison, predictions, confusion, cv_folds, cv_summary)
    mc.rank_models(comparison).to_csv(config.METRICS_DIR / "model_ranking.csv", index=False)
    mc.hybrid_vs_baselines(comparison).to_csv(config.METRICS_DIR / "hybrid_vs_baselines.csv", index=False)
    boot = mc.bootstrap_table(predictions)
    boot.to_csv(config.METRICS_DIR / "bootstrap_hybrid_vs_baselines.csv", index=False)
    if cv_summary is not None:
        mc.write_results_summary(comparison, cv_summary, boot, config.REPORTS_DIR / "results_summary.md")

    step(10, "Model plots, feature importance and SHAP")
    model_plots.generate_all_model_plots(predictions, comparison, confusion, cv_folds)
    feature_importance.run_feature_importance(fitted[config.HYBRID_KEY], X_test, y_test)
    if args.skip_shap:
        logger.info("SHAP skipped (--skip-shap)")
    else:
        run_shap_analysis(fitted[config.HYBRID_KEY], X_test)

    metadata_out = {
        "run_finished": datetime.now().isoformat(timespec="seconds"),
        "python_version": platform.python_version(), "platform": platform.platform(),
        "package_versions": package_versions(), "dataset_file": dataset_path.name,
        "n_rows": int(len(df)), "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "random_state": config.RANDOM_STATE, "cv_folds": config.CV_FOLDS,
        "decision_threshold": config.DECISION_THRESHOLD, "class_weights": config.USE_CLASS_WEIGHTS,
        "hybrid_weights": weights, "cv_run": cv_summary is not None,
    }
    (config.REPORTS_DIR / "run_metadata.json").write_text(json.dumps(metadata_out, indent=2), encoding="utf-8")

    logger.info("Pipeline finished in %.1f s. Start the dashboard with:  streamlit run dashboard/app.py",
                time.time() - start)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except FileNotFoundError as err:
        print(f"\nERROR: {err}")
        sys.exit(1)

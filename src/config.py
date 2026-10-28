"""Central configuration for the churn prediction project.

Every path, seed and modelling switch lives here so that the rest of the code
base never hard-codes them.  All paths use pathlib and are relative to the
project root, so the project runs unchanged on Windows, macOS and Linux.
"""
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_DATA_FILENAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
RAW_DATA_PATH = RAW_DATA_DIR / RAW_DATA_FILENAME

# Public mirrors of the IBM Telco Customer Churn sample dataset (tried in order).
DATASET_URLS = [
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
    "https://raw.githubusercontent.com/treselle-systems/customer_churn_analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv",
]

MODELS_DIR = PROJECT_ROOT / "models"
BASELINE_MODEL_DIR = MODELS_DIR / "baseline"
HYBRID_MODEL_DIR = MODELS_DIR / "hybrid"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessing_pipeline.joblib"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
METRICS_DIR = OUTPUT_DIR / "metrics"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
REPORTS_DIR = OUTPUT_DIR / "reports"

# --------------------------------------------------------------------------- #
# Dataset schema
# --------------------------------------------------------------------------- #
ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"  # raw values: "Yes" / "No"  ->  1 / 0

RAW_FEATURE_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]

# --------------------------------------------------------------------------- #
# Experiment settings
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
TEST_SIZE = 0.20          # 80 % train / 20 % test
CV_FOLDS = 5              # stratified k-fold, training data only
DECISION_THRESHOLD = 0.5  # probability cut-off for the "Churn" class

# Class imbalance handling (Churn is the minority class, ~26 %).  When True,
# LR / DT / RF use class weights and XGBoost uses scale_pos_weight.
USE_CLASS_WEIGHTS = True

# Hybrid soft-voting weights:  P_hybrid = w_rf * P_RF + w_xgb * P_XGB
# (weights are normalised to sum to 1, so (1, 1) is the same as (0.5, 0.5)).
HYBRID_WEIGHTS = {"random_forest": 0.5, "xgboost": 0.5}

# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #
MODEL_NAMES = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "svm": "Support Vector Machine (RBF)",
    "ann": "Artificial Neural Network (MLP)",
    "hybrid": "Hybrid (RF + XGBoost)",
}
HYBRID_KEY = "hybrid"
HYBRID_NAME = MODEL_NAMES[HYBRID_KEY]

MODEL_PATHS = {
    "logistic_regression": BASELINE_MODEL_DIR / "logistic_regression.joblib",
    "decision_tree": BASELINE_MODEL_DIR / "decision_tree.joblib",
    "random_forest": BASELINE_MODEL_DIR / "random_forest.joblib",
    "xgboost": BASELINE_MODEL_DIR / "xgboost.joblib",
    "svm": BASELINE_MODEL_DIR / "svm.joblib",
    "ann": BASELINE_MODEL_DIR / "ann.joblib",
    "hybrid": HYBRID_MODEL_DIR / "rf_xgb_hybrid.joblib",
}

# Columns of the final comparison table (exact order requested by the brief)
METRIC_COLUMNS = [
    "Accuracy", "Precision", "Recall", "Sensitivity", "Specificity", "F1",
    "ROC-AUC", "PR-AUC", "MCC", "Cohen's Kappa", "Log Loss",
]


def ensure_directories() -> None:
    """Create every output directory that the pipeline writes to."""
    for directory in (
        RAW_DATA_DIR, PROCESSED_DATA_DIR, BASELINE_MODEL_DIR, HYBRID_MODEL_DIR,
        FIGURES_DIR, METRICS_DIR, PREDICTIONS_DIR, REPORTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)

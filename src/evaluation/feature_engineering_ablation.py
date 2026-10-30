"""Controlled ablation: compare raw input features against engineered features.

Run from project root:
    python -m src.evaluation.feature_engineering_ablation

The fixed stratified hold-out split is shared with the main project. Preprocessing
is fit only on the training split. This script does not tune on the test set.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score, accuracy_score

from src import config
from src.data.clean_data import clean_raw_dataframe, split_features_target
from src.data.load_data import ensure_dataset, load_raw_data
from src.models.model_training import split_data, make_pipeline
from src.models.baseline_models import build_random_forest


def raw_preprocessor(X):
    numeric = [c for c in ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"] if c in X.columns]
    categorical = [c for c in X.columns if c not in numeric]
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical)
    ], remainder="drop")


def main():
    raw = load_raw_data(ensure_dataset())
    cleaned, _ = clean_raw_dataframe(raw)
    X, y, ids = split_features_target(cleaned)
    X_train, X_test, y_train, y_test, _, _ = split_data(X, y, ids)
    rows = []
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=config.RANDOM_STATE),
        "Random Forest": build_random_forest(),
    }
    for name, estimator in candidates.items():
        raw_pipe = Pipeline([("prep", raw_preprocessor(X_train)), ("model", estimator)])
        raw_pipe.fit(X_train, y_train)
        raw_prob = raw_pipe.predict_proba(X_test)[:, 1]
        raw_pred = (raw_prob >= 0.5).astype(int)
        rows.append({"Model": name, "Feature Set": "Original features", "Accuracy": accuracy_score(y_test, raw_pred),
                     "F1": f1_score(y_test, raw_pred), "Recall": recall_score(y_test, raw_pred),
                     "ROC-AUC": roc_auc_score(y_test, raw_prob), "PR-AUC": average_precision_score(y_test, raw_prob)})
        eng_pipe = make_pipeline(estimator)
        eng_pipe.fit(X_train, y_train)
        eng_prob = eng_pipe.predict_proba(X_test)[:, 1]
        eng_pred = (eng_prob >= 0.5).astype(int)
        rows.append({"Model": name, "Feature Set": "Engineered features", "Accuracy": accuracy_score(y_test, eng_pred),
                     "F1": f1_score(y_test, eng_pred), "Recall": recall_score(y_test, eng_pred),
                     "ROC-AUC": roc_auc_score(y_test, eng_prob), "PR-AUC": average_precision_score(y_test, eng_prob)})
    out = pd.DataFrame(rows)
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.METRICS_DIR / "feature_engineering_ablation.csv", index=False)
    print(out.round(4).to_string(index=False))
    print(f"\nSaved: {config.METRICS_DIR / 'feature_engineering_ablation.csv'}")


if __name__ == "__main__":
    main()

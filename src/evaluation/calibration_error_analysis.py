"""Exploratory calibration and error analysis on saved holdout predictions.

Caution: the project holdout has already been used during model comparison.
These diagnostics are exploratory and are not a new independent validation.
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (brier_score_loss, log_loss, roc_auc_score,
    average_precision_score, precision_score, recall_score, f1_score,
    confusion_matrix, matthews_corrcoef)
from src import config


def expected_calibration_error(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.digitize(p, edges[1:-1], right=False)
    total = len(y)
    return float(sum((idx == b).sum() / total *
        abs(y[idx == b].mean() - p[idx == b].mean())
        for b in range(bins) if (idx == b).any()))


def main():
    base = pd.read_csv(config.PREDICTIONS_DIR / "test_predictions.csv")
    hybrids = pd.read_csv(config.PREDICTIONS_DIR / "hybrid_candidate_test_predictions.csv")
    y = base["y_true"].astype(int).to_numpy()
    probs = {
        "Logistic Regression": base["prob_logistic_regression"].to_numpy(),
        "Random Forest": base["prob_random_forest"].to_numpy(),
        "XGBoost": base["prob_xgboost"].to_numpy(),
        "SVM": base["prob_svm"].to_numpy(),
        "ANN": base["prob_ann"].to_numpy(),
        "Existing RF+XGBoost hybrid": base["prob_hybrid"].to_numpy(),
        "Weighted soft voting": hybrids["prob_weighted_soft_voting_rf_xgb_svm"].to_numpy(),
        "Stacking": hybrids["prob_stacking_rf_xgb_svm_lr"].to_numpy(),
    }
    summary, error_frames = [], []
    for name, p in probs.items():
        pred = (p >= config.DECISION_THRESHOLD).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
        summary.append({
            "Model": name, "ROC-AUC": roc_auc_score(y,p),
            "PR-AUC": average_precision_score(y,p),
            "Brier score (lower better)": brier_score_loss(y,p),
            "Log loss (lower better)": log_loss(y,p,labels=[0,1]),
            "ECE 10 bins (lower better)": expected_calibration_error(y,p),
            "Precision": precision_score(y,pred,zero_division=0),
            "Recall": recall_score(y,pred,zero_division=0),
            "F1": f1_score(y,pred,zero_division=0),
            "MCC": matthews_corrcoef(y,pred),
            "False negatives": int(fn), "False positives": int(fp),
            "True positives": int(tp), "True negatives": int(tn)
        })
        error_frames.append(pd.DataFrame({
            "customerID": base["customer_id"], "y_true": y, "Model": name,
            "predicted_probability": p, "predicted_class": pred,
            "error_type": np.where((y==1)&(pred==0),"false_negative",
                         np.where((y==0)&(pred==1),"false_positive","correct")),
            "absolute_probability_error": np.abs(y-p)
        }))
    metrics = pd.DataFrame(summary).sort_values("PR-AUC", ascending=False)
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(config.METRICS_DIR / "calibration_error_summary.csv", index=False)
    errors = pd.concat(error_frames, ignore_index=True)
    errors.to_csv(config.METRICS_DIR / "per_customer_error_analysis.csv", index=False)

    # Reliability curves for leading probability models
    fig, ax = plt.subplots(figsize=(7,6))
    ax.plot([0,1],[0,1], linestyle=":", label="Perfect calibration")
    for name in ["Existing RF+XGBoost hybrid","Weighted soft voting","Stacking","Logistic Regression"]:
        frac, mean = calibration_curve(y, probs[name], n_bins=10, strategy="quantile")
        ax.plot(mean, frac, marker="o", label=name)
    ax.set(xlabel="Mean predicted churn probability", ylabel="Observed churn fraction",
           title="Reliability curves (exploratory holdout)")
    ax.grid(alpha=.25); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "calibration_reliability.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # High-confidence mistakes are useful audit cases, not proof of a model defect.
    selected = errors[(errors["Model"].isin(["Existing RF+XGBoost hybrid","Weighted soft voting","Stacking"]))
                      & (errors["error_type"] != "correct")].copy()
    selected["confidence_in_wrong_class"] = np.where(
        selected["error_type"]=="false_positive", selected["predicted_probability"],
        1-selected["predicted_probability"])
    selected = selected.sort_values("confidence_in_wrong_class", ascending=False)
    selected.head(100).to_csv(config.REPORTS_DIR / "high_confidence_error_cases.csv", index=False)
    # Subgroup diagnostics on available original fields. These are descriptive
    # and can be unstable for small groups; minimum segment size is 30.
    raw_path = config.RAW_DATA_PATH
    if raw_path.exists():
        raw = pd.read_csv(raw_path)
        raw = raw.rename(columns={config.ID_COLUMN: "customerID"})
        raw["customerID"] = raw["customerID"].astype(str)
        raw["tenure_group"] = pd.cut(raw["tenure"], bins=[-1, 12, 24, 48, 72],
                                     labels=["0-12", "13-24", "25-48", "49-72"])
        raw["monthly_charge_band"] = pd.qcut(raw["MonthlyCharges"], q=4,
                                               duplicates="drop")
        segment_base = raw.merge(base[["customer_id", "y_true"]], left_on="customerID",
                                 right_on="customer_id", how="inner")
        segment_base["y_true"] = segment_base["y_true"].astype(int)
        segment_rows = []
        for model_name, prob in [
            ("Existing RF+XGBoost hybrid", base["prob_hybrid"].to_numpy()),
            ("Weighted soft voting", hybrids["prob_weighted_soft_voting_rf_xgb_svm"].to_numpy()),
            ("Stacking", hybrids["prob_stacking_rf_xgb_svm_lr"].to_numpy())
        ]:
            temp = segment_base.copy()
            temp["prob"] = prob
            temp["pred"] = (temp["prob"] >= config.DECISION_THRESHOLD).astype(int)
            for feature in ["Contract", "InternetService", "tenure_group", "monthly_charge_band"]:
                for group, g in temp.groupby(feature, observed=True):
                    if len(g) < 30:
                        continue
                    positives = int(g["y_true"].sum())
                    tp = int(((g["y_true"] == 1) & (g["pred"] == 1)).sum())
                    fn = int(((g["y_true"] == 1) & (g["pred"] == 0)).sum())
                    fp = int(((g["y_true"] == 0) & (g["pred"] == 1)).sum())
                    segment_rows.append({
                        "Model": model_name, "Segment feature": feature,
                        "Segment": str(group), "Customers": len(g),
                        "Actual churn rate": float(g["y_true"].mean()),
                        "Recall": float(tp/positives) if positives else np.nan,
                        "False negative rate": float(fn/positives) if positives else np.nan,
                        "False positives": fp, "False negatives": fn
                    })
        pd.DataFrame(segment_rows).to_csv(config.METRICS_DIR / "subgroup_error_analysis.csv", index=False)

    note = {
        "purpose":"Exploratory calibration and error analysis",
        "holdout_rows":int(len(y)),
        "threshold":float(config.DECISION_THRESHOLD),
        "metrics":"ROC-AUC, PR-AUC, Brier, log loss, 10-bin ECE, precision, recall, F1, MCC, confusion counts",
        "caveat":"The holdout has been inspected during previous candidate comparison. Results are exploratory, not independent confirmatory evidence.",
        "customer_ids":"Error case export contains customer IDs from the public sample dataset; treat them as row identifiers, not real-world identities."
    }
    (config.REPORTS_DIR / "calibration_error_analysis_metadata.json").write_text(json.dumps(note,indent=2))
    print(metrics[["Model","PR-AUC","Brier score (lower better)","Log loss (lower better)","ECE 10 bins (lower better)","Recall","F1","False negatives"]].round(4).to_string(index=False))
    print("\nSaved calibration metrics, per-customer error table, reliability plot and top error cases.")


if __name__ == "__main__":
    main()

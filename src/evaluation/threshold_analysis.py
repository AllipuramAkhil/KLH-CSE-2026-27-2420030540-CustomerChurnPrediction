"""Decision-threshold analysis for saved hold-out probabilities."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, matthews_corrcoef
from src import config


def main():
    base = pd.read_csv(config.PREDICTIONS_DIR / "test_predictions.csv")
    candidates = pd.read_csv(config.PREDICTIONS_DIR / "hybrid_candidate_test_predictions.csv")
    y = base["y_true"].to_numpy()
    model_probs = {
        "Existing RF+XGBoost hybrid": base["prob_hybrid"].to_numpy(),
        "Stacking": candidates["prob_stacking_rf_xgb_svm_lr"].to_numpy(),
        "Weighted soft voting": candidates["prob_weighted_soft_voting_rf_xgb_svm"].to_numpy(),
    }
    rows = []
    for name, prob in model_probs.items():
        for threshold in np.round(np.arange(0.20, 0.701, 0.025), 3):
            pred = (prob >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
            rows.append({"Model": name, "Threshold": threshold,
                         "Precision": precision_score(y,pred,zero_division=0),
                         "Recall": recall_score(y,pred,zero_division=0),
                         "Specificity": tn/(tn+fp) if tn+fp else np.nan,
                         "F1": f1_score(y,pred,zero_division=0),
                         "MCC": matthews_corrcoef(y,pred),
                         "False Negatives": int(fn), "False Positives": int(fp),
                         "True Positives": int(tp), "True Negatives": int(tn)})
    out = pd.DataFrame(rows)
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.METRICS_DIR / "threshold_analysis.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, group in out.groupby("Model"):
        axes[0].plot(group["Threshold"], group["Recall"], marker=".", label=name)
        axes[0].plot(group["Threshold"], group["Precision"], linestyle="--", alpha=0.8)
        axes[1].plot(group["Threshold"], group["F1"], marker=".", label=name)
    axes[0].set(title="Precision and recall versus threshold", xlabel="Threshold", ylabel="Score")
    axes[1].set(title="F1 versus threshold", xlabel="Threshold", ylabel="F1")
    axes[0].legend(fontsize=7); axes[1].legend(fontsize=7)
    for ax in axes: ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "threshold_analysis.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved threshold table and 300-dpi figure.")
    print(out[out["Threshold"].isin([0.25,0.5,0.65])][["Model","Threshold","Precision","Recall","Specificity","F1","MCC","False Negatives"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()

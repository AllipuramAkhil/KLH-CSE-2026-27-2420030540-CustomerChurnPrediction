"""Paired comparison of hybrid candidate predictions on the fixed hold-out set.

McNemar's exact binomial test compares paired 0/1 error outcomes. Paired bootstrap
resampling estimates a confidence interval for PR-AUC differences. These are
exploratory comparisons and do not replace a pre-registered analysis plan.
"""
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import average_precision_score
from src import config


def mcnemar_exact(y, pred_a, pred_b):
    err_a = np.asarray(pred_a) != np.asarray(y)
    err_b = np.asarray(pred_b) != np.asarray(y)
    a_only = int(np.sum(err_a & ~err_b))
    b_only = int(np.sum(~err_a & err_b))
    discordant = a_only + b_only
    p = float(binomtest(min(a_only, b_only), discordant, 0.5, alternative="two-sided").pvalue) if discordant else 1.0
    return a_only, b_only, discordant, p


def paired_pr_auc_bootstrap(y, p_a, p_b, n_boot=2000, seed=42):
    y = np.asarray(y); p_a = np.asarray(p_a); p_b = np.asarray(p_b)
    rng = np.random.RandomState(seed)
    diffs = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(average_precision_score(y[idx], p_a[idx]) -
                     average_precision_score(y[idx], p_b[idx]))
    point = average_precision_score(y, p_a) - average_precision_score(y, p_b)
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return float(point), float(lo), float(hi)


def main():
    base = pd.read_csv(config.PREDICTIONS_DIR / "test_predictions.csv")
    cand = pd.read_csv(config.PREDICTIONS_DIR / "hybrid_candidate_test_predictions.csv")
    if not np.array_equal(base["y_true"].to_numpy(), cand["y_true"].to_numpy()):
        raise ValueError("Baseline and candidate test labels do not align.")
    y = base["y_true"].to_numpy()
    pairs = [
        ("stacking_rf_xgb_svm_lr", "hybrid", "Stacking vs RF+XGB hybrid"),
        ("weighted_soft_voting_rf_xgb_svm", "hybrid", "Weighted vote vs RF+XGB hybrid"),
        ("weighted_soft_voting_rf_xgb_svm", "xgboost", "Weighted vote vs XGBoost"),
        ("stacking_rf_xgb_svm_lr", "random_forest", "Stacking vs Random Forest"),
    ]
    rows = []
    for candidate, baseline, label in pairs:
        p_c = cand[f"prob_{candidate}"].to_numpy()
        pred_c = cand[f"pred_{candidate}"].to_numpy()
        p_b = base[f"prob_{baseline}"].to_numpy()
        pred_b = base[f"pred_{baseline}"].to_numpy()
        a_only, b_only, discordant, p_mcnemar = mcnemar_exact(y, pred_c, pred_b)
        delta, lo, hi = paired_pr_auc_bootstrap(y, p_c, p_b)
        rows.append({"Comparison": label, "Candidate": candidate, "Reference": baseline,
                     "Candidate errors only": a_only, "Reference errors only": b_only,
                     "Discordant cases": discordant, "McNemar exact p-value": p_mcnemar,
                     "PR-AUC difference (candidate-reference)": delta,
                     "PR-AUC difference 95% bootstrap CI low": lo,
                     "PR-AUC difference 95% bootstrap CI high": hi})
    out = pd.DataFrame(rows)
    out.to_csv(config.METRICS_DIR / "hybrid_paired_significance.csv", index=False)
    print(out.round(5).to_string(index=False))
    print(f"\nSaved: {config.METRICS_DIR / 'hybrid_paired_significance.csv'}")


if __name__ == "__main__":
    main()

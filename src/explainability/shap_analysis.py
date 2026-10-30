"""SHAP explanations for the hybrid model.

The hybrid's output P_hybrid = w1*P_RF + w2*P_XGB is not a single tree ensemble,
so `TreeExplainer` cannot explain it directly.  Instead the model-agnostic
`PermutationExplainer` is applied to the hybrid's churn probability (on the
pre-processed feature matrix), giving SHAP values in probability units for the
hybrid itself.  Values are estimated by sampling, so they are approximate.

SHAP describes what the model associates with higher/lower predicted churn
risk; it does not prove that a feature *causes* churn.

Fallback: if the hybrid explanation fails in a given environment, the XGBoost
member is explained with `TreeExplainer` (log-odds units) and clearly labelled;
if SHAP is unusable altogether the function returns None and the pipeline keeps
the feature-importance outputs.
"""
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)


def _explain_hybrid(pipeline, X_explain, X_background):
    import shap

    pre = pipeline[:-1]
    hybrid = pipeline.named_steps["model"]
    Xt = pre.transform(X_explain)
    Xb = pre.transform(X_background)
    names = list(pre.get_feature_names_out())

    def churn_probability(matrix):
        return hybrid.predict_proba(matrix)[:, 1]

    masker = shap.maskers.Independent(Xb, max_samples=len(Xb))
    explainer = shap.explainers.Permutation(churn_probability, masker, feature_names=names,
                                            seed=config.RANDOM_STATE)
    explanation = explainer(Xt, max_evals=2 * len(names) + 1 + 60, silent=True)
    return np.asarray(explanation.values), Xt, names, "hybrid (probability units)"


def _explain_xgboost_member(pipeline, X_explain):
    import shap

    pre = pipeline[:-1]
    Xt = pre.transform(X_explain)
    names = list(pre.get_feature_names_out())
    values = shap.TreeExplainer(pipeline.named_steps["model"].xgb_).shap_values(Xt)
    return np.asarray(values), Xt, names, "XGBoost member of the hybrid (log-odds units)"


def _save_plots(vals, Xt, names, label):
    import shap

    importance = (pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(vals).mean(axis=0),
                                "explained_model": label})
                  .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))
    importance.to_csv(config.METRICS_DIR / "shap_importance.csv", index=False)

    plt.figure(figsize=(9, 7))
    shap.summary_plot(vals, Xt, feature_names=names, max_display=15, show=False)
    plt.title(f"SHAP summary - {label}", fontsize=10)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "13_shap_summary_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    plt.figure(figsize=(9, 6))
    shap.summary_plot(vals, Xt, feature_names=names, max_display=15, plot_type="bar", show=False)
    plt.title(f"Global importance (mean |SHAP|) - {label}", fontsize=10)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "13b_shap_importance_bar.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    return importance


def run_shap_analysis(pipeline, X_test, n_explain=150, n_background=40):
    """Save SHAP summary + bar plots and mean |SHAP| CSV.  Returns the importance table or None."""
    rng = np.random.RandomState(config.RANDOM_STATE)
    explain = X_test.iloc[rng.choice(len(X_test), size=min(n_explain, len(X_test)), replace=False)]
    background = X_test.iloc[rng.choice(len(X_test), size=min(n_background, len(X_test)), replace=False)]
    for attempt in (lambda: _explain_hybrid(pipeline, explain, background),
                    lambda: _explain_xgboost_member(pipeline, explain)):
        try:
            vals, Xt, names, label = attempt()
            importance = _save_plots(vals, Xt, names, label)
            logger.info("SHAP analysis saved (%s).", label)
            return importance
        except Exception as exc:  # noqa: BLE001 - SHAP/version problems must not break the pipeline
            plt.close("all")
            logger.warning("SHAP attempt failed (%s).", exc)
    logger.warning("SHAP unavailable; using feature-importance outputs instead.")
    return None

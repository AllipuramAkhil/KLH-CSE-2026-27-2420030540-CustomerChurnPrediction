"""Feature importance: model-internal (RF / XGBoost) and permutation-based."""
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance

from src import config

logger = logging.getLogger(__name__)


def transformed_feature_names(pipeline) -> list[str]:
    return list(pipeline[:-1].get_feature_names_out())


def tree_based_importance(hybrid_pipeline) -> pd.DataFrame:
    """Impurity importance (RF) and gain importance (XGBoost) of the hybrid's members."""
    names = transformed_feature_names(hybrid_pipeline)
    hybrid = hybrid_pipeline.named_steps["model"]
    df = pd.DataFrame({"feature": names,
                       "rf_importance": hybrid.rf_.feature_importances_,
                       "xgb_importance": hybrid.xgb_.feature_importances_})
    w_rf, w_xgb = hybrid._normalised_weights()
    df["hybrid_importance"] = w_rf * df["rf_importance"] + w_xgb * df["xgb_importance"]
    return df.sort_values("hybrid_importance", ascending=False).reset_index(drop=True)


def permutation_importance_raw(pipeline, X_test, y_test, n_repeats=10) -> pd.DataFrame:
    """Drop in ROC-AUC when one ORIGINAL input column is shuffled (works on raw columns).

    Computed on the test set for *explanation* only - it is never used to choose a model.
    """
    result = permutation_importance(pipeline, X_test, y_test, scoring="roc_auc", n_repeats=n_repeats,
                                    random_state=config.RANDOM_STATE, n_jobs=1)
    df = pd.DataFrame({"feature": X_test.columns, "importance_mean": result.importances_mean,
                       "importance_std": result.importances_std})
    return df.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def plot_tree_importance(df: pd.DataFrame, top_n=15):
    top = df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["feature"], top["hybrid_importance"], color="#E45756")
    ax.set(title=f"Top {top_n} features by hybrid tree-based importance\n(weighted average of RF and XGBoost importances)",
           xlabel="Importance (relative)")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "12_feature_importance_tree_based.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_permutation_importance(df: pd.DataFrame, top_n=12):
    top = df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="#4C78A8")
    ax.set(title="Permutation importance of the hybrid model (original input columns)",
           xlabel="Decrease in test ROC-AUC when the column is shuffled")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "12b_permutation_importance_hybrid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_feature_importance(hybrid_pipeline, X_test, y_test):
    config.ensure_directories()
    tree_df = tree_based_importance(hybrid_pipeline)
    perm_df = permutation_importance_raw(hybrid_pipeline, X_test, y_test)
    tree_df.to_csv(config.METRICS_DIR / "feature_importance_tree_based.csv", index=False)
    perm_df.to_csv(config.METRICS_DIR / "feature_importance_permutation.csv", index=False)
    plot_tree_importance(tree_df)
    plot_permutation_importance(perm_df)
    return tree_df, perm_df

# Proposed model: Random Forest + XGBoost weighted soft voting

## Formula
```
P_hybrid(churn) = w_rf * P_RF(churn) + w_xgb * P_XGB(churn),      w_rf + w_xgb = 1
predicted class = 1 if P_hybrid >= 0.5 else 0
```
Default `w_rf = w_xgb = 0.5` (`HYBRID_WEIGHTS` in `src/config.py`, or `python run.py --rf-weight 0.6 --xgb-weight 0.4`; weights are normalised).

## Implementation
`WeightedSoftVotingClassifier` in `src/models/hybrid_model.py` is a real scikit-learn estimator:
1. `fit` clones and trains a Random Forest and an XGBoost classifier on the same preprocessed training data.
2. `predict_proba` returns the weighted average of the two probability vectors.
3. `predict` applies the decision threshold.
It sits inside the same Pipeline as the baselines (feature engineering -> preprocessing -> classifier), so it takes raw customer records and is cross-validated exactly like the other models. Its two members use the **same hyper-parameters** as the stand-alone Random Forest and XGBoost baselines, so the comparison isolates the effect of combining them.
Notebook `05_hybrid_model.ipynb` verifies numerically that `P_hybrid == 0.5*P_RF + 0.5*P_XGB`.

## Why these two models
| | Random Forest | XGBoost |
|---|---|---|
| Family | Bagging: many trees trained independently on bootstrap samples, predictions averaged | Boosting: trees trained sequentially, each correcting the errors of the previous ones |
| Main effect | Reduces variance | Reduces bias step by step (with regularisation) |
| Strengths | Non-linear relationships, robust to noisy features | Strong on structured/tabular data, captures complex interactions |
Averaging two models built on different learning strategies *can* give more stable predictions, because their errors are not perfectly correlated.

## What the experiment actually showed
On this dataset the two members are very similar: their churn probabilities have a correlation of about 0.98 on the test set and they assign a different class to only about 6 % of customers. When members are that alike, averaging has little to correct, so gains are necessarily small. See `docs/model_comparison.md` for the measured results and the honest verdict.
A weight sweep on out-of-fold training predictions (notebook 05) changed CV ROC-AUC only in the third decimal place, so the simple equal-weight ensemble was kept and no weight was tuned on the test set.

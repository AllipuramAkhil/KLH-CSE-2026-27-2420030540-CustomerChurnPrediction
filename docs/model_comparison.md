# Model comparison - results and honest interpretation

Numbers below come from the run stored in `outputs/` (Python 3.12.3, seed 42). Re-running `python run.py` regenerates them together with `outputs/reports/results_summary.md`; small differences are possible with other library versions.

## Test set (1,409 customers, threshold 0.5)
| Model | Acc. | Prec. | Recall/Sens. | Spec. | F1 | ROC-AUC | PR-AUC | MCC | Kappa | Log Loss |
|---|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.7353 | 0.5008 | **0.7888** | 0.7159 | 0.6127 | 0.8420 | 0.6370 | 0.4518 | 0.4264 | 0.4950 |
| Decision Tree | 0.7488 | 0.5181 | 0.7674 | 0.7420 | 0.6185 | 0.8321 | 0.6244 | 0.4605 | 0.4415 | 0.5673 |
| Random Forest | **0.7693** | **0.5476** | 0.7540 | **0.7749** | 0.6344 | 0.8413 | 0.6462 | 0.4850 | 0.4721 | **0.4623** |
| XGBoost | 0.7559 | 0.5271 | 0.7807 | 0.7469 | 0.6293 | 0.8422 | **0.6551** | 0.4770 | 0.4573 | 0.4774 |
| **Hybrid (RF + XGBoost)** | 0.7672 | 0.5434 | 0.7701 | 0.7662 | **0.6372** | **0.8437** | 0.6529 | **0.4888** | **0.4732** | 0.4661 |

Hybrid confusion matrix: TN 793, FP 242, FN 86, TP 288.

## 5-fold CV on the training set (mean +/- std)
| Model | ROC-AUC | PR-AUC | F1 | MCC |
|---|---|---|---|---|
| Logistic Regression | 0.8449 +/- 0.0133 | 0.6610 +/- 0.0139 | 0.6259 +/- 0.0231 | 0.4725 +/- 0.0362 |
| Decision Tree | 0.8318 +/- 0.0138 | 0.6176 +/- 0.0306 | 0.6179 +/- 0.0098 | 0.4603 +/- 0.0150 |
| Random Forest | 0.8459 +/- 0.0102 | 0.6590 +/- 0.0240 | 0.6319 +/- 0.0157 | 0.4820 +/- 0.0225 |
| XGBoost | 0.8402 +/- 0.0102 | 0.6520 +/- 0.0217 | 0.6280 +/- 0.0144 | 0.4753 +/- 0.0214 |
| Hybrid (RF + XGBoost) | 0.8452 +/- 0.0101 | 0.6588 +/- 0.0233 | 0.6322 +/- 0.0142 | 0.4820 +/- 0.0206 |

## Verdict (what the evidence supports)
1. **The hybrid is competitive, not clearly superior.** On the test set it has the highest ROC-AUC, F1, MCC and Cohen's Kappa, but it is *not* best on PR-AUC (XGBoost), Log Loss, Accuracy, Precision and Specificity (Random Forest) or Recall (Logistic Regression).
2. **In cross-validation the picture is slightly different**: Random Forest has the highest mean ROC-AUC (0.8459 vs 0.8452 for the hybrid) and Logistic Regression the highest PR-AUC (0.6610). The hybrid ties Random Forest on mean MCC and F1.
3. **Most gaps are inside the noise.** The CV standard deviations (about 0.010 for ROC-AUC) are several times larger than the gaps between the top models, and the paired-bootstrap 95 % intervals for hybrid minus Logistic Regression, Random Forest and XGBoost include 0 for both ROC-AUC and PR-AUC. Only the improvement over the Decision Tree is statistically clear (ROC-AUC +0.0116, 95 % CI +0.0009 to +0.0223).
4. **The single Decision Tree is the weakest model**: it has the highest Log Loss and the lowest ROC-AUC and PR-AUC on both the test set and in cross-validation (only the hybrid-vs-tree difference was tested with the bootstrap).
5. **A simple Logistic Regression is very strong** on this dataset (ROC-AUC 0.842 on test), which may indicate that much of the churn signal is roughly additive and leaves little room for complex models to add value (this was not tested separately).

## Trade-offs to discuss
- Recall vs. precision: with class weighting, every model catches roughly 75-79 % of churners but only about 50-55 % of flagged customers actually churn. Which model is "best" depends on the relative cost of a missed churner vs. a wasted retention offer.
- Log Loss is inflated by class weighting (probabilities are shifted upward); calibration would be needed before treating the probability as a literal risk percentage.
- All conclusions are for one dataset and one random split (plus CV); they should not be generalised without further data.

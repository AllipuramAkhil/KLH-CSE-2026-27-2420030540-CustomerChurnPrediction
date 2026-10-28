# Review-3 content (ready to paste into slides)

> The project **was executed**, so real values are filled in (seed 42, Python 3.12.3, test set = 1,409 customers). If you re-run on another machine, compare with `outputs/reports/results_summary.md` and update any number that differs. Nothing here is a claim that the hybrid is "the best": the wording follows the measured results.

---
## 1. Problem Statement
- Telecom companies lose revenue when customers leave (churn); keeping a customer is cheaper than acquiring a new one.
- Goal: predict which customers are likely to churn so retention offers can be targeted.
- Formulation: **binary classification**, 1 = churn, 0 = stay; classes are imbalanced (26.5 % churn).
- Faculty requirement: the proposed approach must be more than one basic model -> **Random Forest + XGBoost hybrid ensemble** compared against four baselines.

## 2. Dataset and Data Sources
- IBM Telco Customer Churn sample dataset (also on Kaggle: *blastchar/telco-customer-churn*).
- 7,043 customers, 19 input features + `customerID` + target `Churn`.
- Feature groups: demographics (gender, senior citizen, partner, dependents), services (phone, multiple lines, internet, security, backup, device protection, tech support, streaming TV/movies), account (tenure, contract, paperless billing, payment method, monthly and total charges).
- Churn rate 26.54 % (1,869 churners / 5,174 stayers).

## 3. Data Preprocessing and Cleaning
- `TotalCharges` was text; 11 blank values (all `tenure = 0`, not yet billed) converted to numeric 0.0.
- 0 exact duplicates, 0 duplicate IDs; `customerID` excluded from features; `Churn` Yes/No -> 1/0.
- No outliers by the 1.5 x IQR rule in tenure, MonthlyCharges, TotalCharges; no rows removed.
- Median imputation + standardisation (numeric) and mode imputation + one-hot (categorical) **inside an sklearn Pipeline** -> fitted on training data only (no leakage).
- Stratified 80/20 split (5,634 train / 1,409 test), `random_state = 42`; test set untouched until final evaluation.

## 4. Exploratory Data Analysis
| Finding | Churn rate |
|---|---|
| Month-to-month / one-year / two-year contract | 42.7 % / 11.3 % / 2.8 % |
| Fiber optic / DSL / no internet | 41.9 % / 19.0 % / 7.4 % |
| Electronic check vs. other payment methods | 45.3 % vs 15.2-19.1 % |
| Tenure 0-12 months / 61+ months | 47.4 % / 6.6 % |
| Senior citizens / non-seniors | 41.7 % / 23.6 % |
| Month-to-month **and** fiber optic (interaction) | 54.6 % |
- Median tenure: churners 10 months vs. stayers 38; median monthly charge: churners 79.65 vs. stayers 64.43.
- Figures: `outputs/figures/01`-`06c`. These are associations, not proof of cause.

## 5. Baseline Models
| Model | Why included | Main settings |
|---|---|---|
| Logistic Regression | simple, interpretable linear benchmark | C = 1, class-weight balanced |
| Decision Tree | interpretable non-linear rules | depth 5, min leaf 20 |
| Random Forest | bagging ensemble, lower variance | 300 trees, depth 10, min leaf 5 |
| XGBoost | boosting ensemble, strong on tabular data | 300 trees, lr 0.05, depth 4, subsample 0.8 |
- 5-fold stratified cross-validation on the training set; no heavy tuning (academic comparison).

## 6. Feature Engineering
12 features: `tenure_group`, `service_count`, `protection_service_count`, `streaming_service_count`, `avg_monthly_charge`, `charge_delta`, `charges_per_service`, `is_month_to_month`, `auto_payment`, `is_electronic_check`, `has_family`, `mtm_fiber` (month-to-month x fiber optic). Each is computed row-by-row (stateless), so it is leakage-safe. Definitions and reasons: `docs/methodology.md`.

## 7. Proposed Hybrid Model
- **Random Forest + XGBoost weighted soft voting.**
- `P_hybrid = 0.5 x P_RF + 0.5 x P_XGB`; class = churn if `P_hybrid >= 0.5`; weights configurable.
- Implemented as a scikit-learn estimator inside the same Pipeline; both members use the same preprocessing and the same hyper-parameters as the stand-alone baselines.

## 8. Hybrid Architecture
```
Raw customer record
   -> Feature engineering (12 features)
   -> Imputation / scaling / one-hot (fitted on train only)
        |-> Random Forest  -> P_RF  --|
        |                             |--> 0.5*P_RF + 0.5*P_XGB = P_hybrid --> threshold 0.5 --> Churn / No churn
        |-> XGBoost        -> P_XGB --|
```
Bagging (variance reduction, independent trees) + boosting (sequential error correction, captures complex interactions).

## 9. Evaluation Metrics
Accuracy, Precision, Recall = Sensitivity (TP/(TP+FN)), Specificity (TN/(TN+FP)), F1, ROC-AUC, PR-AUC, MCC, Cohen's Kappa, Log Loss, confusion matrix. Positive class = churn. Accuracy alone is misleading because 73.5 % of customers do not churn (predicting "no churn" for everyone gives 73.5 % accuracy and 0 recall).

## 10. Comparative Performance Analysis
**Test set**
| Model | Acc. | Prec. | Recall | Spec. | F1 | ROC-AUC | PR-AUC | MCC | Kappa | Log Loss |
|---|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.7353 | 0.5008 | 0.7888 | 0.7159 | 0.6127 | 0.8420 | 0.6370 | 0.4518 | 0.4264 | 0.4950 |
| Decision Tree | 0.7488 | 0.5181 | 0.7674 | 0.7420 | 0.6185 | 0.8321 | 0.6244 | 0.4605 | 0.4415 | 0.5673 |
| Random Forest | 0.7693 | 0.5476 | 0.7540 | 0.7749 | 0.6344 | 0.8413 | 0.6462 | 0.4850 | 0.4721 | 0.4623 |
| XGBoost | 0.7559 | 0.5271 | 0.7807 | 0.7469 | 0.6293 | 0.8422 | 0.6551 | 0.4770 | 0.4573 | 0.4774 |
| Hybrid | 0.7672 | 0.5434 | 0.7701 | 0.7662 | 0.6372 | 0.8437 | 0.6529 | 0.4888 | 0.4732 | 0.4661 |

**5-fold CV (train), ROC-AUC mean +/- std:** LR 0.8449 +/- 0.0133; DT 0.8318 +/- 0.0138; RF 0.8459 +/- 0.0102; XGB 0.8402 +/- 0.0102; Hybrid 0.8452 +/- 0.0101.
Charts: `07_roc_curves`, `08_precision_recall_curves`, `09_model_comparison_auc_f1_mcc`, `10_model_comparison_accuracy_precision_recall_specificity`, `11_confusion_matrix_*`, `14_cv_rocauc_boxplot`.

## 11. Results and Discussion
- Hybrid confusion matrix (test): TN 793, FP 242, FN 86, TP 288 -> catches 288 of 374 churners (77.0 %); 242 of 1,035 stayers are flagged unnecessarily.
- The hybrid has the **highest ROC-AUC (0.8437), F1 (0.6372), MCC (0.4888) and Kappa (0.4732)** on the test set.
- It is **not** best on PR-AUC (XGBoost 0.6551), Log Loss / Accuracy / Precision / Specificity (Random Forest) or Recall (Logistic Regression 0.7888).
- In cross-validation, Random Forest has a marginally higher mean ROC-AUC (0.8459) than the hybrid (0.8452).
- Differences among LR, RF, XGB and the hybrid are much smaller than the fold-to-fold standard deviation (~0.01). Paired-bootstrap 95 % CIs of hybrid-minus-baseline ROC-AUC: vs LR [-0.0066, +0.0101], vs RF [-0.0007, +0.0056], vs XGB [-0.0017, +0.0047] (all include 0); vs Decision Tree [+0.0009, +0.0223] (excludes 0).
- Conclusion: the hybrid is a **competitive and stable** model and clearly better than a single Decision Tree, but the experiment does **not** show a significant advantage over Random Forest, XGBoost or Logistic Regression.

## 12. Justification for Hybrid Model
- Different learning strategies (bagging vs boosting) can make errors partly independent; averaging probabilities can smooth them and gives a single, stable score.
- It was chosen *before* seeing results, as required by the faculty; results are reported as measured.
- Why the gain is small: on this dataset RF and XGB probabilities are ~0.98 correlated and disagree on only ~6 % of customers, so there is little disagreement for the ensemble to exploit.
- Practical benefits that do not depend on winning every metric: lower variance than a single boosted model, no dependence on one algorithm's quirks, one configurable weight.

## 13. Key Insights
Model-associated drivers (SHAP and permutation importance agree): **contract type, tenure, fiber-optic internet/`mtm_fiber`, total charges, lack of online security and tech support, electronic-check payment**.
- Permutation importance (ROC-AUC drop): Contract 0.077, tenure 0.044, InternetService 0.015, TotalCharges 0.008.
- Customers on month-to-month contracts with short tenure and fiber internet form the highest-risk segment (54.6 % churn for month-to-month + fiber).
- Business reading (association, not causation): target retention offers at new month-to-month customers, encourage longer contracts and automatic payments, and bundle security/support services.

## 14. Limitations
Single public dataset and single split (plus CV); no time dimension; class weighting makes probabilities uncalibrated and the 0.5 threshold is not cost-based; small differences between top models; SHAP is a sampling-based approximation; associations only.

## 15. Future Scope
Probability calibration; cost-based threshold; nested-CV hyper-parameter tuning; stacking with a meta-learner; time-based validation; fairness checks; drift monitoring; API deployment.

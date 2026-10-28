# Methodology

## 1. Pipeline overview
```
raw CSV -> cleaning (rule-based) -> stratified 80/20 split
        -> [ feature engineering -> imputation/scaling/one-hot -> classifier ]   (one sklearn Pipeline)
        -> 5-fold stratified CV on TRAIN  -> final fit on TRAIN -> one-time evaluation on TEST
```

## 2. Cleaning (`src/data/clean_data.py`)
| Issue | Treatment |
|---|---|
| `TotalCharges` stored as text with blank strings | `pd.to_numeric`; the 11 blanks all have `tenure = 0` and are set to 0.0 (business rule) |
| Duplicates | Exact duplicate rows are dropped (0 found). Duplicate IDs: 0. 40 rows share identical *features* with another customer; they are different customers and are kept |
| Identifier | `customerID` is never a feature |
| Target | `Yes/No` -> `1/0` |
| Constant columns | Checked programmatically (none) |
| Outliers | IQR rule reported in `outputs/metrics/outlier_report_iqr.csv`: 0 outliers in tenure, MonthlyCharges, TotalCharges, so nothing is removed |
| Missing values | None remain after cleaning; the pipeline still contains imputers so that new data with gaps is handled |

Cleaning only applies fixed rules. Everything that is *learned* from data (median, mean, std, category lists) happens inside the Pipeline.

## 3. Leakage prevention
Data leakage = information from the test set (or the future) influencing training.
- The split happens **before** any fitting; `stratify=y`, `random_state=42`.
- Imputer, scaler and one-hot encoder are inside the Pipeline, so they are fitted on training rows only - and re-fitted on the training part of each CV fold.
- Feature engineering is stateless (each new value uses only the same row).
- The test set is used once, for the final report. Model selection and the hybrid weight check use cross-validation on the training set only.
- Permutation importance and SHAP use test rows only to *explain* the finished model, never to choose it.
- `scale_pos_weight` for XGBoost is computed from the training labels only.

## 4. Feature engineering (`src/features/feature_engineering.py`)
| Feature | Type | Definition | Why |
|---|---|---|---|
| `tenure_group` | categorical | 0-12, 13-24, 25-48, 49-60, 61+ months | churn falls non-linearly with tenure |
| `service_count` | numeric | number of subscribed services (phone, multiple lines, internet, 6 add-ons) | engagement / lock-in |
| `protection_service_count` | numeric | OnlineSecurity + OnlineBackup + DeviceProtection + TechSupport | support-type services are associated with lower churn |
| `streaming_service_count` | numeric | StreamingTV + StreamingMovies | entertainment usage |
| `avg_monthly_charge` | numeric | TotalCharges / tenure (MonthlyCharges when tenure = 0) | historical average bill |
| `charge_delta` | numeric | MonthlyCharges - avg_monthly_charge | current bill vs. history (price-change signal) |
| `charges_per_service` | numeric | MonthlyCharges / max(service_count, 1) | price paid per service |
| `is_month_to_month` | binary | Contract == Month-to-month | no commitment |
| `auto_payment` | binary | bank transfer or credit card (automatic) | payment friction |
| `is_electronic_check` | binary | PaymentMethod == Electronic check | segment with highest observed churn |
| `has_family` | binary | Partner or Dependents = Yes | household stability |
| `mtm_fiber` | binary | month-to-month AND fiber optic | interaction of two high-risk factors seen in EDA |

After one-hot encoding the models see 61 input columns.
Several engineered features overlap with raw ones (e.g. `is_month_to_month` and `Contract_Month-to-month`). That is harmless for trees and regularised logistic regression, but it means importance is split across correlated columns; permutation importance on the *original* columns is provided for a cleaner reading.

## 5. Splitting and cross-validation
- 80 % train (5,634 rows) / 20 % test (1,409 rows), stratified; churn rate 26.54 % in both.
- 5-fold `StratifiedKFold(shuffle=True, random_state=42)` on the training set for every model (including the hybrid); mean and standard deviation of all metrics are saved in `outputs/metrics/cross_validation_summary.csv`.

## 6. Class imbalance
About 26.5 % of customers churn. Logistic Regression, Decision Tree and Random Forest use class weights (`balanced` / `balanced_subsample`), XGBoost uses `scale_pos_weight = n_neg / n_pos` (switch off with `USE_CLASS_WEIGHTS = False` in `src/config.py`).
Consequences: recall rises and precision falls compared with an unweighted model at a 0.5 threshold, and the predicted probabilities are **not calibrated** (they over-state the churn probability). Log Loss values must be read with that in mind.

## 7. Metrics
Positive class = churn = 1. Computed from real predictions with sklearn (`src/evaluation/metrics.py`).
| Metric | Meaning for churn |
|---|---|
| Accuracy | share of all customers classified correctly (misleading with 26 % churners) |
| Precision | of customers flagged as churners, the share that really churn |
| Recall = Sensitivity = TP/(TP+FN) | of real churners, the share the model catches |
| Specificity = TN/(TN+FP) | of real non-churners, the share correctly left alone |
| F1 | harmonic mean of precision and recall |
| ROC-AUC | probability that a random churner gets a higher score than a random non-churner (threshold-free) |
| PR-AUC (average precision) | area under precision-recall curve; more informative when the positive class is rare |
| MCC | correlation between prediction and truth using all four confusion-matrix cells; -1 to 1 |
| Cohen's Kappa | agreement beyond chance |
| Log Loss | penalises confident wrong probabilities; lower is better |

## 8. Statistical caution
Differences between the models are small. `outputs/metrics/bootstrap_hybrid_vs_baselines.csv` gives paired-bootstrap 95 % intervals (1,000 resamples of the test set) for hybrid minus baseline ROC-AUC and PR-AUC. An interval containing 0 means the difference cannot be separated from sampling noise on this test set.

## 9. Explainability
- Tree-based importance (RF impurity + XGBoost gain, averaged with the hybrid weights).
- Permutation importance of the hybrid on the original input columns (drop in ROC-AUC).
- SHAP (`PermutationExplainer`) applied to the hybrid's probability on 150 test rows with 40 background rows. It is a sampling-based approximation. `TreeExplainer` cannot explain the blended model directly, and its interventional mode failed with this XGBoost/SHAP version combination, hence the model-agnostic explainer.
All statements are associations ("associated with higher predicted churn risk"), never causal claims.

## 10. Reproducibility
`random_state = 42` everywhere. Environment and package versions of each run are stored in `outputs/reports/run_metadata.json`. Results can differ slightly across library versions or CPU/thread settings.

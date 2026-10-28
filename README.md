# Intelligent Customer Churn Prediction
**Using Data Analytics and Hybrid Machine Learning** - semester project for *Data Analytics and Visualization (DAV)*.

An end-to-end, reproducible churn-prediction system: cleaning, EDA, feature engineering, four baseline models, a proposed **Random Forest + XGBoost soft-voting hybrid**, 12 evaluation metrics, explainability (SHAP) and a 6-page Streamlit dashboard with single and batch prediction.

> **Scope of claims.** The hybrid is compared honestly against its baselines. In the reported run it is competitive but **not clearly superior** (see section 13). Nothing in this repository is hard-coded: every table and chart is computed from real predictions.

---

## 1. Project Overview
The pipeline (`python run.py`) loads the IBM Telco Customer Churn dataset, cleans it, splits it 80/20 (stratified), cross-validates and trains five models inside leakage-safe scikit-learn pipelines, evaluates them on the untouched test set, saves models/metrics/figures, and the dashboard (`streamlit run dashboard/app.py`) presents everything and scores new customers.

## 2. Problem Statement
Acquiring a customer costs more than retaining one. Given a customer's demographics, services and billing information, predict whether they will churn so retention offers can be targeted. This is **binary classification** (1 = churn, 0 = stay) on imbalanced data (~26.5 % churn).

## 3. Objectives
1. Clean and explore the data; identify churn drivers.
2. Engineer meaningful features.
3. Train Logistic Regression, Decision Tree, Random Forest and XGBoost baselines.
4. Build a genuine Random Forest + XGBoost hybrid ensemble.
5. Evaluate with 11 metrics + confusion matrix, cross-validation and bootstrap intervals.
6. Explain predictions (feature importance, permutation importance, SHAP).
7. Deliver an interactive dashboard with batch scoring.

## 4. Dataset
IBM Telco Customer Churn: 7,043 customers, 19 input features, target `Churn` (Yes/No). See `data/README.md` for sources and the `TotalCharges` quirk. The CSV is included in `data/raw/` and downloaded automatically if missing.

## 5. Technologies Used
Python 3.12 (3.11+), pandas, NumPy, scikit-learn, XGBoost, SHAP, matplotlib, seaborn, Plotly, Streamlit, joblib. Exact versions in `requirements.txt`.

## 6. Project Architecture
```
customer-churn-dav/
├── data/            raw/ (original CSV), processed/ (generated), README.md
├── notebooks/       01..06 step-by-step notebooks (already executed)
├── src/
│   ├── config.py                    paths, seed, weights, switches
│   ├── data/                        load_data, clean_data, preprocess
│   ├── eda/                         exploratory_analysis
│   ├── features/                    feature_engineering (ChurnFeatureEngineer)
│   ├── models/                      baseline_models, hybrid_model, model_training
│   ├── evaluation/                  metrics, evaluate_models, model_comparison
│   ├── visualization/               eda_plots, model_plots, feature_importance
│   └── explainability/              shap_analysis
├── dashboard/       app.py + components/ (data_access, prediction)
├── models/          baseline/*.joblib, hybrid/rf_xgb_hybrid.joblib, preprocessing_pipeline.joblib
├── outputs/         figures/, metrics/, predictions/, reports/
├── docs/            methodology, hybrid_model, model_comparison, review3_content, review3_viva_questions
├── run.py           end-to-end pipeline
└── requirements.txt
```
Every model is one `Pipeline`: `ChurnFeatureEngineer -> ColumnTransformer(impute/scale/one-hot) -> classifier`, so saved models accept **raw customer columns**.

## 7. Data Preprocessing
Rule-based cleaning (`TotalCharges` to numeric, duplicates, identifier removal, target encoding) followed by learned preprocessing (median imputation + standard scaling for numerics; mode imputation + one-hot for categoricals) **inside the pipeline**, fitted on training data only. Details: `docs/methodology.md`.

## 8. EDA
Selected findings (associations, not causes):
- Churn rate by contract: month-to-month 42.7 %, one year 11.3 %, two year 2.8 %.
- Fiber-optic customers 41.9 % vs DSL 19.0 % and no internet 7.4 %.
- Electronic-check payers 45.3 % vs 15-19 % for other methods.
- Tenure 0-12 months: 47.4 %; 61+ months: 6.6 %.
- Senior citizens 41.7 % vs 23.6 %.
Figures are in `outputs/figures/` (01-06c).

## 9. Feature Engineering
Twelve features (tenure group, service counts, average monthly charge, charge delta, charges per service, month-to-month flag, auto-payment flag, electronic-check flag, family flag, month-to-month x fiber interaction). Each is documented in `src/features/feature_engineering.py` and `docs/methodology.md`.

## 10. Baseline Models
Logistic Regression, Decision Tree (depth 5, min leaf 20), Random Forest (300 trees, depth 10, min leaf 5), XGBoost (300 trees, lr 0.05, depth 4, subsample 0.8, colsample 0.8). Class imbalance handled with class weights / `scale_pos_weight`. Fixed sensible parameters - no heavy tuning.

## 11. Proposed Hybrid Model
`P_hybrid = w_rf * P_RF + w_xgb * P_XGB` (default 0.5 / 0.5), implemented as `WeightedSoftVotingClassifier` (`src/models/hybrid_model.py`). Weights are configurable in `src/config.py` or via `python run.py --rf-weight 0.6 --xgb-weight 0.4`. See `docs/hybrid_model.md`.

## 12. Evaluation Metrics
Accuracy, Precision, Recall, Sensitivity, Specificity, F1, ROC-AUC, PR-AUC, MCC, Cohen's Kappa, Log Loss, and the confusion matrix. **Positive class = churn**: Recall/Sensitivity = share of real churners caught; Specificity = share of real non-churners correctly identified. Also 5-fold stratified CV (mean and std) on the training set, and paired-bootstrap intervals for the hybrid-vs-baseline differences.

## 13. Results
Test set (seed 42, 1,409 customers). Full tables: `outputs/metrics/`, auto-generated `outputs/reports/results_summary.md`, interpretation in `docs/model_comparison.md`.

| Model | Recall | Specificity | F1 | ROC-AUC | PR-AUC | MCC | Log Loss |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.7888 | 0.7159 | 0.6127 | 0.8420 | 0.6370 | 0.4518 | 0.4950 |
| Decision Tree | 0.7674 | 0.7420 | 0.6185 | 0.8321 | 0.6244 | 0.4605 | 0.5673 |
| Random Forest | 0.7540 | 0.7749 | 0.6344 | 0.8413 | 0.6462 | 0.4850 | 0.4623 |
| XGBoost | 0.7807 | 0.7469 | 0.6293 | 0.8422 | 0.6551 | 0.4770 | 0.4774 |
| **Hybrid (RF + XGB)** | 0.7701 | 0.7662 | 0.6372 | 0.8437 | 0.6529 | 0.4888 | 0.4661 |

**Honest reading:** the hybrid has the highest ROC-AUC, F1, MCC and Kappa on the test set but not the best PR-AUC, Log Loss, Recall, Precision, Specificity or Accuracy; in cross-validation Random Forest has a marginally higher mean ROC-AUC (0.8459 vs 0.8452). Differences among the top models are far smaller than the fold-to-fold standard deviation (~0.01), and bootstrap intervals against LR, RF and XGBoost include zero. Only the gain over the single Decision Tree is statistically clear.
Top model-associated drivers (SHAP / permutation importance): contract type, tenure, fiber-optic internet, total charges, missing online security / tech support, electronic-check payment.

## 14. Dashboard
`streamlit run dashboard/app.py` - six pages: Overview, EDA, Model Comparison (hybrid highlighted, no superiority claim), Model Evaluation (confusion matrix, ROC, PR, metrics per model), Churn Prediction (form + batch CSV) and Explainability. The dashboard loads saved models and results; it never retrains.

## 15. Installation
```bash
python -m venv .venv
# Windows (PowerShell / cmd):
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```
Requires Python 3.11+ and no GPU.

## 16. Running the Project
```bash
python run.py                 # ~1.5 min on a laptop: clean, CV, train, evaluate, save, plot, SHAP
streamlit run dashboard/app.py
```
Options: `python run.py --skip-shap` (faster), `--skip-cv`, `--rf-weight 0.6 --xgb-weight 0.4`.
The ZIP already contains trained models and results from one run, so the dashboard works immediately; if a model file cannot be loaded because of a library-version difference, the dashboard tells you to run `python run.py` to recreate it in your environment.
Notebooks: `jupyter notebook notebooks/` (run `run.py` first for notebook 06).

## 17. Batch Prediction
Dashboard -> *Churn Prediction* -> *Batch (CSV)*. Upload a CSV with the 19 feature columns (a template can be downloaded on that page; `customerID` optional). The app validates the columns, warns about unknown category values, applies the same preprocessing, and lets you download `customer_id, churn_probability, churn_prediction`.

## 18. Project Limitations
- One dataset, one random split (plus CV); no external validation. Results may not transfer to other telecoms or time periods.
- Static snapshot: no time dimension, so "will churn" is really "resembles customers who churned".
- Class weighting raises recall at the cost of precision and makes probabilities uncalibrated; the 0.5 threshold is not cost-optimised.
- The hybrid's members are very similar (probability correlation ~0.98), so its gain over its components is small and not statistically significant.
- SHAP values are sampling-based approximations on 150 test rows; importance is shared across correlated/engineered features. Associations are not causal effects.
- No hyper-parameter search by design (academic comparison rather than leaderboard tuning).

## 19. Future Scope
Probability calibration; cost-sensitive threshold selection using retention cost/benefit; hyper-parameter tuning with nested CV; stacking with a meta-learner (including logistic regression); time-based validation on real customer histories; fairness analysis across groups; monitoring/drift detection; deployment as an API.

## 20. Conclusion
The project delivers a complete, leakage-safe, reproducible churn-prediction workflow. Ensemble and linear models reach a ROC-AUC of about 0.84 on this dataset. The proposed RF + XGBoost hybrid is a competitive, stable option and leads on several balanced metrics on the test set, but the evidence does not show it to be significantly better than Random Forest, XGBoost or Logistic Regression. The main practical findings are about *who* churns (month-to-month, short-tenure, fiber-optic, electronic-check customers), which is more valuable to a retention team than the small differences between models.

---
Reproducibility details of the reported run: `outputs/reports/run_metadata.json`. Python 3.12.3, seed 42.


## Root-level Streamlit entry point

From the project root, launch the dashboard with:

```bash
streamlit run app.py
```

The root `app.py` delegates to `dashboard/app.py`. Before launching, install the dependencies in `requirements.txt` and run `python run.py --skip-shap` at least once to create the model and evaluation artifacts. For a faster smoke test, use `python run.py --skip-cv --skip-shap`.


## Added baselines and feature ablation

The model registry now includes Logistic Regression, Decision Tree, Random Forest, XGBoost, RBF SVM, an MLP-based ANN, and the existing RF/XGBoost soft-voting hybrid. A controlled feature ablation can be run with:

```bash
python -m src.evaluation.feature_engineering_ablation
```

See `docs/STAGE3_BASELINES_REPORT.md` for the current recorded results and execution limitations. Results are specific to the current Python/package environment and should be regenerated before final publication.


## Hybrid candidate experiments (Stage 4)

Five hybrid candidates are implemented and compared using training-only stratified five-fold CV and the fixed hold-out split:

- RF + XGBoost + SVM stacking with a Logistic Regression meta-learner
- Weighted soft voting across RF, XGBoost and SVM
- RF feature selection + ANN
- XGBoost leaf representations + ANN
- Chi-square feature selection + RF/XGBoost ensemble

Run all candidates with:

```bash
python -m src.evaluation.run_hybrid_candidates
```

Run or refresh a single candidate and merge its outputs with the existing candidate results:

```bash
python -m src.evaluation.run_hybrid_candidates --only stacking_rf_xgb_svm_lr
```

Run paired McNemar and bootstrap comparisons:

```bash
python -m src.evaluation.hybrid_significance
```

See `docs/STAGE4_HYBRID_RESEARCH_REPORT.md` and `docs/hybrid_candidate_architectures.md`. Current results do not establish a statistically significant PR-AUC improvement; do not claim a universal winner or novelty without literature verification.


## Stage 5: tuning and threshold analysis

Training-only hyperparameter tuning (stratified 5-fold `RandomizedSearchCV`, PR-AUC refit) and threshold analysis are implemented:

```bash
python -m src.evaluation.tune_hybrid_candidates
python -m src.evaluation.threshold_analysis
```

The tuning run did not finish in the constrained execution environment, so no tuned results are claimed yet. See `docs/STAGE5_TUNING_REPORT.md`.


## Stage 6: calibration and error analysis

Run the exploratory calibration, error, and subgroup diagnostics on the saved holdout predictions:

```bash
python -m src.evaluation.calibration_error_analysis
```

Outputs include calibration metrics, a reliability plot, per-customer errors, high-confidence error cases, and subgroup diagnostics. See `docs/STAGE6_CALIBRATION_ERROR_REPORT.md`. The previously examined holdout is exploratory, not untouched confirmatory evidence.


## Stage 7: literature register and reproducibility audit

- Literature register: `docs/VERIFIED_LITERATURE_REGISTER.md`
- Audit and threshold interpretation: `docs/STAGE7_REPRODUCIBILITY_AUDIT.md`
- Threshold analysis: `python -m src.evaluation.threshold_analysis`

The Stage 5 tuning search is implemented but has not completed in the constrained runtime. Streamlit launch and SHAP execution remain unverified in that runtime. See the Stage 7 report for limitations and reproducibility commands.


## Stage 8: research paper draft

- Editable Word paper: `docs/STAGE8_RESEARCH_PAPER_DRAFT.docx`
- Markdown source: `docs/STAGE8_RESEARCH_PAPER_DRAFT.md`
- Stage report and submission checklist: `docs/STAGE8_RESEARCH_PAPER_REPORT.md`

This is a provisional draft, not a final submission. The paper explicitly discloses that Stage 5 tuning is incomplete, SHAP execution is unverified, and the previously inspected holdout is exploratory. Replace the author placeholder and complete the outstanding validation before submitting.

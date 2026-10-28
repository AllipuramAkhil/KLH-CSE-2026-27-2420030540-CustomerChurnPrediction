# Comparative Evaluation of Hybrid Models for Telecom Customer Churn Prediction: Discrimination, Calibration, and Error Analysis

**Research paper draft — Stage 8**  
**Author:** [Add author name and affiliation]  
**Status:** Provisional academic draft; hyperparameter tuning and SHAP execution remain incomplete.

## Abstract

Customer churn prediction is a common classification task in subscription businesses, but evaluation by accuracy alone can obscure the trade-off between detecting departing customers and avoiding unnecessary retention actions. This study compares conventional classifiers and five hybrid candidates on the commonly circulated IBM Telco Customer Churn dataset containing 7,043 customer records. The workflow uses a stratified 80/20 train–test split with random seed 42, preprocessing pipelines, and five-fold stratified cross-validation on the training partition. Models include Logistic Regression, Decision Tree, Random Forest, XGBoost, radial-basis-function Support Vector Machine, a multilayer perceptron, a Random Forest–XGBoost probability ensemble, stacking, weighted soft voting, feature-selection hybrids, and an XGBoost leaf-representation neural model. Performance is assessed using ROC-AUC, precision–recall AUC (PR-AUC), F1, Matthews correlation coefficient, Brier score, log loss, and threshold-based error counts. In training cross-validation, Random Forest achieved mean ROC-AUC 0.8459, while Logistic Regression achieved mean PR-AUC 0.6610. On the previously inspected holdout, weighted soft voting achieved PR-AUC 0.6573 among the five new hybrid candidates, while stacking achieved F1 0.6387. Paired bootstrap intervals for PR-AUC differences included zero, and multiple-comparison-adjusted McNemar tests did not establish a significant advantage. The analysis illustrates that model rankings depend on the chosen objective: weighted voting had stronger PR-AUC and probability scores than the existing two-model hybrid, while stacking detected more positive cases at the fixed 0.50 threshold. These results are exploratory because the holdout was repeatedly examined during candidate development. Hyperparameter tuning has not completed, and SHAP execution has not been verified; therefore, no final superiority, deployment-readiness, or completed explainability claim is made.

**Keywords:** customer churn, telecommunications, classification, hybrid ensemble, stacking, soft voting, PR-AUC, calibration, error analysis, explainable AI.

## I. Introduction

Customer churn occurs when a customer ends or does not renew a service relationship. In subscription-based industries, churn prediction can help organizations prioritize customer-retention outreach. However, churn models are only useful when their evaluation reflects the costs of different mistakes. A false negative leaves an at-risk customer unidentified, while a false positive may lead to unnecessary outreach or incentives.

Machine-learning research on churn commonly includes linear classifiers, decision trees, random forests, and gradient-boosting methods. A recent systematic review of churn prediction research reports continued use of ensemble methods and identifies recurring concerns such as class imbalance, interpretability, concept drift, and limited business-oriented evaluation [1]. Research on telecom churn has also explored post-hoc explanation methods, including SHAP, to make model outputs more interpretable [2].

This project investigates a practical comparison of six individual classifiers and multiple hybrid candidates using a public telecom benchmark. Rather than selecting a model from accuracy alone, it reports threshold-independent discrimination metrics, threshold-dependent classification measures, probability-quality scores, and exploratory error diagnostics.

The paper addresses three questions:

1. How do the individual baseline models compare under training-only cross-validation and a fixed holdout evaluation?
2. Do the tested hybrid candidates show a consistent advantage over the existing Random Forest–XGBoost probability ensemble?
3. How do threshold choice and probability calibration change the interpretation of model performance?

The intended contribution is a reproducible, multi-metric comparison on the chosen benchmark. It is not a claim that the hybrid architecture is the first of its kind or that the results generalize to production telecom customers.

## II. Related Work

### A. Churn prediction models

Churn prediction has been studied across telecommunications, banking, retail, and other subscription or relationship-based services. Imani et al. review machine-learning and deep-learning studies published from 2020 through 2024 and describe ensemble learning, class imbalance, interpretability, and business-aligned metrics as recurring themes [1]. Because datasets, churn definitions, feature sets, and split protocols differ across studies, published headline metrics should not be directly ranked without accounting for those differences.

### B. Explainability

Lundberg and Lee introduced SHAP as a unified framework for assigning feature-attribution values to individual predictions [3]. Poudel et al. applied explainability techniques to telecom churn models and examined local and global explanations alongside predictive performance [2]. Such explanations describe how a fitted model uses features; they do not, by themselves, establish that a feature causes churn.

The present codebase includes a SHAP workflow with a model-agnostic explanation path for the probability ensemble and a fallback for its XGBoost member. However, generated SHAP artifacts were not found in the Stage 7 audit. Consequently, this draft does not present SHAP rankings or explanation plots as completed empirical findings.

### C. Evaluation beyond accuracy

ROC-AUC summarizes ranking performance across thresholds, whereas PR-AUC emphasizes the precision–recall trade-off and is useful when the positive class is less common. F1 summarizes precision and recall at one threshold but does not reflect true negatives. Probability quality is assessed separately using Brier score, log loss, and expected calibration error (ECE). These metrics are not interchangeable; a model can have good discrimination but poorly calibrated probabilities, or good calibration while missing many churners. Calibration guidance and leakage-prevention practices are documented by scikit-learn [4], [5].

## III. Dataset and Preprocessing

### A. Dataset

The experiment uses the commonly circulated IBM Telco Customer Churn CSV (`WA_Fn-UseC_-Telco-Customer-Churn.csv`), containing 7,043 records and 21 raw columns. The target field is `Churn`, encoded as 1 for “Yes” and 0 for “No.” The observed churn prevalence is approximately 26.54%. The CSV is available in IBM’s archived public repository [6]. The dataset describes a fictional telecommunications setting and should not be interpreted as current data from a real operator.

**Reproducibility note:** record the exact download URL and retrieval date in the final submission. The local file SHA-256 at the time of this draft is: `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`. The raw dataset contains customer identifiers; these are used for row alignment and are not model predictors.

### B. Cleaning and features

The project’s cleaning pipeline handles the target encoding and numeric parsing required by the raw file. The feature pipeline performs feature engineering and preprocessing within a scikit-learn pipeline. This design allows learned preprocessing transformations to be fitted separately within each cross-validation training fold. The input includes demographic, service, contract, payment, tenure, and charge variables. Customer ID is excluded from the predictor matrix.

### C. Data partition

The project uses an 80/20 stratified train–test split with random seed 42. The training partition is used for five-fold stratified cross-validation. Model pipelines keep imputation, encoding, scaling, and feature selection within the training folds where applicable.

A key limitation must be made explicit: the same holdout was examined repeatedly during hybrid candidate development and calibration/error analysis. Although the Stage 5 parameter search code uses training-only cross-validation, the overall holdout is no longer an untouched confirmatory test set. The holdout results below are therefore reported as exploratory. Confirmatory evaluation requires a fresh independent dataset or a pre-specified nested cross-validation design.

## IV. Methods

### A. Baseline classifiers

The individual model set contains Logistic Regression, Decision Tree, Random Forest, XGBoost, radial-basis-function SVM, and a multilayer perceptron neural network. A two-model probability ensemble combines Random Forest and XGBoost. Baseline hyperparameters are configured in the project source and are not represented as exhaustive tuning.

### B. Hybrid candidates

Five candidate hybrids were implemented:

1. **Stacking:** Random Forest, XGBoost, and SVM base estimators, with Logistic Regression as the meta-learner trained on cross-validated base predictions.
2. **Weighted soft voting:** a probability ensemble of Random Forest, XGBoost, and SVM with weights 0.2, 0.6, and 0.2, respectively.
3. **Random Forest feature selection plus ANN:** a Random Forest-based selector followed by an MLP classifier.
4. **XGBoost leaf representation plus ANN:** encoded terminal-leaf representations from a boosting model are passed to an MLP.
5. **Chi-square feature selection plus ensemble:** scaled encoded features are filtered by chi-square selection before a Random Forest/XGBoost voting ensemble.

The hybrid search and comparison are documented in the project’s Stage 4 report. Hyperparameter tuning of stacking and weighted voting is implemented in code but has not completed in the available execution environment. No tuned values are included in this paper.

### C. Metrics

The study reports:
- **Accuracy:** fraction of all observations classified correctly.
- **Precision:** fraction of predicted churners who actually churned.
- **Recall:** fraction of actual churners detected.
- **F1:** harmonic mean of precision and recall.
- **ROC-AUC:** ranking performance over classification thresholds.
- **PR-AUC:** area under the precision–recall curve.
- **MCC:** a balanced summary based on all four confusion-matrix cells.
- **Brier score and log loss:** probabilistic prediction quality, with lower values preferred.
- **ECE:** a 10-bin descriptive calibration summary, with lower values preferred.

All classification metrics using a binary prediction use the project’s fixed 0.50 probability threshold unless otherwise stated.

### D. Statistical comparison

The Stage 4 analysis used exact McNemar tests on paired holdout classifications and paired bootstrap intervals for PR-AUC differences. Because several comparisons were considered, a Bonferroni threshold of 0.0125 was used for the four McNemar comparisons. These tests do not prove equivalence; they indicate that the available sample did not establish a reliable difference under the specified analysis. Repeated inspection of the holdout limits the strength of confirmatory inference.

### E. Calibration and threshold analysis

Brier score, log loss, and ECE were computed from saved holdout probabilities. Threshold diagnostics compared 0.25, 0.50, and 0.65 as illustrative operating points. These thresholds were not optimized as an independent business policy. In a deployment study, threshold choice should be made using validation data and explicit costs for missed churners, false alarms, and retention offers.

## V. Results

### A. Baseline comparison

Table I reports holdout results for the individual models and the existing two-model hybrid. The MLP achieved the highest accuracy (0.7977) but had recall of 0.5588. Logistic Regression achieved recall of 0.7888 and PR-AUC of 0.6370. Random Forest reached F1 of 0.6344, while the existing Random Forest–XGBoost hybrid reached ROC-AUC 0.8439 and PR-AUC 0.6510.

**Table I. Baseline model results on the exploratory holdout**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7353 | 0.5008 | 0.7888 | 0.6127 | 0.8420 | 0.6370 |
| Decision Tree | 0.7488 | 0.5181 | 0.7674 | 0.6185 | 0.8321 | 0.6244 |
| Random Forest | 0.7693 | 0.5476 | 0.7540 | 0.6344 | 0.8413 | 0.6462 |
| XGBoost | 0.7530 | 0.5233 | 0.7807 | 0.6266 | 0.8428 | 0.6498 |
| RBF SVM | 0.7828 | 0.5833 | 0.6364 | 0.6087 | 0.8249 | 0.6142 |
| ANN (MLP) | 0.7977 | 0.6353 | 0.5588 | 0.5946 | 0.8365 | 0.6230 |
| RF + XGBoost hybrid | 0.7644 | 0.5396 | 0.7647 | 0.6327 | 0.8439 | 0.6510 |

### B. Training cross-validation

In five-fold stratified cross-validation on the training partition, Random Forest had mean ROC-AUC 0.8459 (standard deviation 0.0102). Logistic Regression had mean PR-AUC 0.6610 (standard deviation 0.0139), slightly above Random Forest’s mean PR-AUC of 0.6590 (standard deviation 0.0240). The RF+XGBoost hybrid had mean ROC-AUC 0.8450 and mean PR-AUC 0.6582. The results show that different metrics produce different rankings; no model dominates every metric.

### C. Hybrid candidate comparison

**Table II. Five new hybrid candidates**

| Candidate | CV ROC-AUC mean | CV PR-AUC mean | Holdout F1 | Holdout ROC-AUC | Holdout PR-AUC |
|---|---:|---:|---:|---:|---:|
| Stacking RF + XGB + SVM + LR | 0.8455 | 0.6586 | 0.6387 | 0.8419 | 0.6464 |
| Weighted soft voting RF + XGB + SVM | 0.8440 | 0.6577 | 0.6253 | 0.8443 | 0.6573 |
| Chi-square feature-selection ensemble | 0.8402 | 0.6448 | 0.6216 | 0.8342 | 0.6464 |
| RF selection + ANN | 0.8399 | 0.6436 | 0.5714 | 0.8406 | 0.6379 |
| XGB leaf representation + ANN | 0.8311 | 0.6384 | 0.5662 | 0.8291 | 0.6206 |

Stacking achieved the highest holdout F1 among these five candidates, while weighted soft voting achieved the highest holdout ROC-AUC and PR-AUC. The absolute PR-AUC difference between weighted voting and the existing RF+XGBoost hybrid was +0.0063, with a paired 95% bootstrap interval from −0.0005 to +0.0130. The interval includes zero. The stacking versus existing hybrid PR-AUC difference was −0.0046, with a 95% interval from −0.0110 to +0.0014.

For the four McNemar comparisons in Stage 4, none remained significant after applying the Bonferroni threshold of 0.0125. The result is insufficient evidence of a statistically reliable winner, not proof that models are identical.

### D. Calibration

**Table III. Selected probability-quality results**

| Model | PR-AUC | Brier score ↓ | Log loss ↓ | ECE ↓ |
|---|---:|---:|---:|---:|
| Weighted soft voting | 0.6573 | 0.1496 | 0.4508 | 0.0964 |
| RF + XGBoost hybrid | 0.6510 | 0.1555 | 0.4665 | 0.1171 |
| Stacking | 0.6464 | 0.1663 | 0.5041 | 0.1566 |
| ANN (MLP) | 0.6230 | 0.1401 | 0.4267 | 0.0244 |

Weighted soft voting had better Brier score, log loss, and ECE than the existing two-model hybrid in this exploratory evaluation. The MLP had the lowest Brier score and ECE, but its PR-AUC and recall were lower. These differences illustrate why calibration and churn-detection performance should be considered separately. ECE depends on binning and sample size; it should not be treated as a definitive measure on its own.

### E. Threshold sensitivity

For stacking, lowering the threshold from 0.50 to 0.25 increased recall from 0.7941 to 0.9278 and reduced false negatives from 77 to 27. Precision fell from 0.5342 to 0.4242, while specificity fell from 0.7498 to 0.5449. At a threshold of 0.65, recall fell to 0.6444 and specificity increased to 0.8300. This is the expected trade-off between catching more churners and limiting false alarms.

## VI. Discussion

The findings do not support a single universal best model. The MLP maximized accuracy in the holdout but missed more churners than the higher-recall models. Weighted soft voting ranked best on holdout PR-AUC among the five new hybrids and had more favorable probability-quality scores than the existing two-model hybrid. Stacking yielded the strongest F1 among the new candidates and detected more positives at the 0.50 threshold, but its probability calibration was less favorable.

For a retention team, the preferred model depends on the cost of missed churners, the budget for outreach, the success rate of retention offers, and the acceptable false-positive volume. Those business costs were not provided in the current dataset; therefore, the study does not calculate financial return or claim that any threshold is optimal.

The results are consistent with the broader literature’s recommendation to evaluate churn models using more than a single accuracy value [1]. Calibration and error analysis add useful diagnostic views, but they do not substitute for temporal validation, external replication, or an intervention study measuring whether targeted retention actually reduces churn.

## VII. Limitations

1. **Previously examined holdout:** The 20% holdout was reused across model comparison, hybrid development, and diagnostics. Its scores are exploratory and should not be presented as independent final validation.
2. **Tuning incomplete:** The randomized hyperparameter search was implemented but did not complete in the available execution environment. The reported results are from the pre-tuning candidates.
3. **Explainability incomplete:** The SHAP module exists, but no generated SHAP artifacts were found in the audit. No empirical SHAP feature ranking is claimed here.
4. **Dashboard not launch-tested:** Streamlit is listed in the requirements but was not installed in the audit runtime. Interactive dashboard operation still needs verification.
5. **Static public dataset:** The benchmark is a sample dataset and does not capture current customer behavior, time drift, interventions, or operational constraints.
6. **No business cost model:** The study does not specify contact costs, offer costs, customer lifetime value, or retention uplift. It cannot identify a profit-maximizing threshold.
7. **Subgroup analysis is descriptive:** Segment-level error differences require uncertainty estimates and independent validation before any fairness or operational claim.
8. **Literature review is not exhaustive:** The source register is a starting point. The paper does not claim a systematic review or first-of-its-kind method.

## VIII. Conclusion and Future Work

This project compared individual classifiers and five hybrid candidates for telecom churn prediction using a reproducible pipeline on a 7,043-record public dataset. Cross-validation and exploratory holdout results show trade-offs among discrimination, classification threshold, and probability calibration. Weighted soft voting produced the highest holdout PR-AUC among the five new hybrid candidates, while stacking achieved the highest holdout F1 among those candidates. However, paired uncertainty intervals included zero, and multiple-comparison-adjusted tests did not establish significant superiority.

Future work should complete the training-only hyperparameter search, predefine the operating threshold using business costs, validate on a fresh or temporally later dataset, verify SHAP outputs, and launch-test the dashboard in a clean environment. Only after those steps should the project make final model-selection or deployment claims.

## References

[1] M. Imani, M. Joudaki, A. Beikmohammadi, and H. R. Arabnia, “Customer Churn Prediction: A Systematic Review of Recent Advances, Trends, and Challenges in Machine Learning and Deep Learning,” *Machine Learning and Knowledge Extraction*, vol. 7, no. 3, Art. no. 105, Sep. 2025, doi: 10.3390/make7030105.

[2] S. S. Poudel, S. Pokharel, and M. Timilsina, “Explaining customer churn prediction in telecom industry using tabular machine learning models,” *Machine Learning with Applications*, vol. 17, Art. no. 100567, Sep. 2024, doi: 10.1016/j.mlwa.2024.100567.

[3] S. M. Lundberg and S.-I. Lee, “A Unified Approach to Interpreting Model Predictions,” in *Advances in Neural Information Processing Systems 30 (NeurIPS 2017)*, 2017.

[4] scikit-learn developers, “Probability calibration,” scikit-learn documentation. Available: https://scikit-learn.org/stable/modules/calibration.html. Accessed: Oct. 25, 2026.

[5] scikit-learn developers, “Common pitfalls and recommended practices,” scikit-learn documentation. Available: https://scikit-learn.org/stable/common_pitfalls.html. Accessed: Oct. 25, 2026.

[6] IBM, “Telco Customer Churn,” IBM/telco-customer-churn-on-icp4d, archived public repository. Available: https://github.com/IBM/telco-customer-churn-on-icp4d. Accessed: Oct. 25, 2026.

## Appendix A. Reproduction commands

From the project root, create a clean Python 3.12 environment, install `requirements.txt`, and run:

```bash
python run.py
python -m src.evaluation.run_hybrid_candidates
python -m src.evaluation.calibration_error_analysis
python -m src.evaluation.threshold_analysis
python -m src.evaluation.tune_hybrid_candidates
streamlit run app.py
```

The full pipeline and tuning can take substantial time. Verify outputs after each command and record the package versions, dataset source and checksum, random seed, and any failures. Do not state that SHAP or Streamlit ran successfully until the corresponding outputs or successful launch have been verified.

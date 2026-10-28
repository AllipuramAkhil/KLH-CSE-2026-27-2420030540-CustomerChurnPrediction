# Stage 4 Report: Hybrid Candidate Experiments

## Scope and protocol
Five distinct hybrid candidates were implemented and compared on the prescribed IBM Telco Customer Churn CSV. All candidates use the same 80/20 stratified hold-out split (seed 42). Candidate model selection uses five-fold stratified CV on training data only. Preprocessing, feature selection and learned encodings are inside sklearn pipelines and are fitted separately within each CV training fold.

Candidate stacking uses RF, XGBoost and RBF SVM base learners, with Logistic Regression as a meta-learner trained on out-of-fold predictions. To avoid leaking the inner stacking validation fold's class prevalence into XGBoost's configuration, the XGBoost member in stacking uses `scale_pos_weight=1.0`; RF and SVM still use their class-weight behavior. Other candidate XGBoost weights are derived from the outer fold's training labels.

## Five hybrid candidates
1. **Stacking RF + XGBoost + SVM + Logistic Regression meta-learner.** Diverse model probabilities are combined by a learned meta-model.
2. **Weighted soft voting RF + XGBoost + SVM.** Probabilities are combined using fixed weights RF=0.2, XGBoost=0.6, SVM=0.2. Weights have not been tuned yet.
3. **RF feature selection + ANN.** RF importance selects features before an MLP neural classifier.
4. **XGBoost leaf representations + ANN.** Each record is represented by one-hot encoded terminal leaf indices from boosting trees and passed to an MLP.
5. **Chi-square feature selection + RF/XGBoost ensemble.** Min-max scaled encoded features are ranked by chi-square; the top 25 feed a soft-voting ensemble.

See `docs/hybrid_candidate_architectures.md` for Mermaid diagrams.

## Five-fold stratified CV and hold-out results

| Candidate | CV ROC-AUC mean | CV ROC-AUC SD | CV PR-AUC mean | CV F1 mean | Test accuracy | Test F1 | Test ROC-AUC | Test PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| stacking_rf_xgb_svm_lr | 0.8455 | 0.0103 | 0.6586 | 0.6298 | 0.7615 | 0.6387 | 0.8419 | 0.6464 |
| weighted_soft_voting_rf_xgb_svm | 0.8440 | 0.0123 | 0.6577 | 0.6276 | 0.7644 | 0.6253 | 0.8443 | 0.6573 |
| chi2_feature_selection_ensemble | 0.8402 | 0.0127 | 0.6448 | 0.6323 | 0.7537 | 0.6216 | 0.8342 | 0.6464 |
| rf_feature_selection_ann | 0.8399 | 0.0103 | 0.6436 | 0.5786 | 0.7956 | 0.5714 | 0.8406 | 0.6379 |
| xgb_leaf_embedding_ann | 0.8311 | 0.0155 | 0.6384 | 0.5578 | 0.7814 | 0.5662 | 0.8291 | 0.6206 |

Reference results from the current baseline hold-out evaluation:

| Reference model | Test accuracy | Test F1 | Test ROC-AUC | Test PR-AUC |
|---|---:|---:|---:|---:|
| Random Forest | 0.7693 | 0.6344 | 0.8413 | 0.6462 |
| XGBoost | 0.7530 | 0.6266 | 0.8428 | 0.6498 |
| Hybrid (RF + XGBoost) | 0.7644 | 0.6327 | 0.8439 | 0.6510 |

## Interpretation
- **Highest mean CV ROC-AUC:** stacking, 0.8455 (SD 0.0103). This is only marginally above the existing RF+XGBoost hybrid's recorded CV ROC-AUC mean of approximately 0.8450; that small difference is not evidence of a meaningful improvement by itself.
- **Highest candidate hold-out PR-AUC:** weighted soft voting, 0.6573. It is above the existing RF+XGBoost hybrid's 0.6510 in this test split, but uncertainty must be considered.
- **Highest candidate hold-out F1:** stacking, 0.6387. This is close to Random Forest's 0.6344 and does not establish universal superiority.
- **RF selection + ANN** and **XGBoost leaves + ANN** are weaker on F1 in this configuration. These are useful negative results; the architectures may require tuning, but no improvement should be claimed.
- **Chi-square selection + ensemble** has the highest mean CV F1 among these five candidates (0.6323), but lower hold-out ROC-AUC than the stronger alternatives.

## Paired significance analysis

The analysis uses the same held-out customer rows for each paired comparison. McNemar's exact binomial test compares hard-classification errors. A paired bootstrap estimates the PR-AUC difference and its 95% percentile interval using 2,000 resamples.

| Comparison | McNemar exact p | PR-AUC difference | 95% bootstrap interval |
|---|---:|---:|---:|
| Stacking vs RF+XGB hybrid | 0.6076 | -0.0046 | [-0.0110, +0.0014] |
| Weighted vote vs RF+XGB hybrid | 1.0000 | +0.0063 | [-0.0005, +0.0130] |
| Weighted vote vs XGBoost | 0.0293 | +0.0075 | [-0.0024, +0.0191] |
| Stacking vs Random Forest | 0.1173 | +0.0002 | [-0.0042, +0.0039] |

**Statistical conclusion:** all paired bootstrap 95% intervals for PR-AUC differences include zero. The weighted-vote versus XGBoost McNemar p-value is 0.0293 before correction, but there are four comparisons; a Bonferroni threshold is 0.0125, so none of the McNemar results is significant after this simple multiplicity correction. Do not claim a statistically significant performance improvement from these tests.

## Provisional model-selection decision

Do not declare a final research winner yet. The results support carrying **weighted soft voting** and **stacking** into the tuning stage as the two leading candidates: weighted voting has the best candidate hold-out PR-AUC, while stacking has the highest mean CV ROC-AUC and best candidate hold-out F1. Tune hyperparameters and voting weights using training-only CV, then make one final decision under a predeclared primary metric. The held-out test set has already been inspected for this exploration, so the final paper must acknowledge that model selection has used these test results; ideally, reserve a fresh final evaluation protocol if feasible or present this test set as exploratory rather than an untouched confirmatory set.

## Novelty statement and literature caveat

These are combinations of established methods. This report does **not** claim that any candidate is the first such combination or globally novel. A defensible contribution may be a careful, leakage-safe comparative study and a dataset-specific design choice. A novelty claim requires a verified literature review and comparison of the exact combination, training protocol, feature representation and objective.

## Reproducibility notes and limitations
- Candidate CV folds: `outputs/metrics/hybrid_candidate_cv_folds.csv`.
- Candidate summary: `outputs/metrics/hybrid_candidate_comparison.csv`.
- Candidate predictions: `outputs/predictions/hybrid_candidate_test_predictions.csv`.
- Paired significance table: `outputs/metrics/hybrid_paired_significance.csv`.
- Run metadata: `outputs/reports/hybrid_candidate_run_metadata.json`.
- Experiment runner: `python -m src.evaluation.run_hybrid_candidates`.
- Rerun one candidate and merge it into the existing outputs with `python -m src.evaluation.run_hybrid_candidates --only stacking_rf_xgb_svm_lr`.
- Paired tests: `python -m src.evaluation.hybrid_significance`.
- Hyperparameter tuning is not yet implemented.
- These measurements are specific to this environment and configuration. Re-run the full experiment after dependency versions are fixed before using the results in a paper.
# Stage 5 Report: Training-only tuning and decision thresholds

## Implemented in this stage
- Added `src/evaluation/tune_hybrid_candidates.py`, using `RandomizedSearchCV`, stratified five-fold CV and average precision (PR-AUC) as the refit metric.
- The search space covers stacking meta-learner regularization, RF tree count/depth, XGBoost tree count/depth/learning rate, SVM regularization, and several predeclared weighted-voting configurations.
- The search uses only the training partition. It only evaluates the hold-out partition after each candidate's best configuration has been selected by training CV.
- Added `src/evaluation/threshold_analysis.py` to compare precision, recall, specificity, F1, MCC and false-negative counts over thresholds from 0.20 to 0.70 for the existing hybrid, stacking and weighted voting.
- Candidate architectures and paired statistical tests from Stage 4 remain available.

## Execution status
The tuning command was started but did not complete in this execution environment. It entered the first candidate's five-fold, eight-draw search (`40 fits`) and was interrupted by the environment's execution-time limit before saving a tuning summary. Therefore:
- **No tuned metrics are reported in this stage.**
- No best hyperparameters are claimed.
- The tuning script should be run on the student's local machine or a longer-running environment to complete.
- The threshold-analysis script is implemented but was not run after the tuning attempt.

Run from the project root:

```bash
python -m src.evaluation.tune_hybrid_candidates
python -m src.evaluation.threshold_analysis
```

The tuning script saves `outputs/metrics/hybrid_tuning_summary.csv`, `outputs/metrics/hybrid_tuning_cv_results.csv`, `outputs/reports/hybrid_tuning_metadata.json`, and tuned model artifacts when it completes. The threshold script saves `outputs/metrics/threshold_analysis.csv` and `outputs/figures/threshold_analysis.png` at 300 dpi.

## Important methodological caveat
The fixed hold-out set has already been examined during hybrid candidate exploration. Even though tuning itself uses training-only cross-validation and only accesses the hold-out after selecting a configuration, this test set is no longer a fully untouched confirmatory set for the overall research project. The paper should disclose this. If possible, use a new final evaluation split or nested CV before making confirmatory claims.

## Current evidence-based status
The leading untuned candidates from Stage 4 were stacking and weighted soft voting. Weighted voting had the highest hold-out PR-AUC among the five candidates (0.6573); stacking had the highest candidate mean CV ROC-AUC (0.8455) and hold-out F1 (0.6387). Paired bootstrap confidence intervals for PR-AUC differences included zero, and no McNemar comparison remained significant after the simple Bonferroni correction. These are exploratory findings, not proof of superiority.

## Next steps
1. Run the tuning and threshold analysis commands to completion.
2. Add calibration evaluation and error-case explanations.
3. Verify SHAP and Streamlit in a clean installation.
4. Verify literature references before making novelty claims or writing final publication claims.

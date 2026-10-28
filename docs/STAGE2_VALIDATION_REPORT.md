# Stage 2: Reproducibility and Pipeline Validation

## Scope

This report describes validation of the isolated working copy. The uploaded source ZIP and the prescribed IBM Telco CSV were not modified. The active working copy contains a root-level `app.py`, updated dependency pins, and a revised cross-validation implementation.

## Completed fixes

1. Replaced `cross_validate` with an explicit stratified-fold loop in `src/models/model_training.py` so the XGBoost class-imbalance ratio is computed from each fold's training labels only. The hybrid's XGBoost component receives the same fold-local override.
2. Added root `app.py` to support the requested command `streamlit run app.py`; it delegates to `dashboard/app.py`.
3. Updated `requirements.txt` to match versions observed in the validation runtime where possible. Streamlit remains declared as a dependency, but it was not installed in the validation runtime, so installation and dashboard launch remain unverified.
4. Updated README launch instructions.
5. Regenerated five-fold CV fold and summary CSV/JSON outputs after the code change.

## Runtime validation

- `python -m compileall -q /mnt/data/customer-churn-dav-stage1`: passed.
- `python run.py --skip-cv --skip-shap`: completed successfully in 30.5 seconds.
- The run loaded the prescribed dataset (7,043 rows, 21 columns), cleaned it, generated EDA, trained four baselines and the existing RF/XGBoost hybrid, evaluated the 1,409-row hold-out set, saved model artifacts and metrics, and generated model plots.
- XGBoost and hybrid fold-weight smoke tests: passed. Both estimators were fit on a training subset with a fold-local ratio, and predictions were finite and within [0, 1].
- Full five-fold CV for all five existing model pipelines: passed in 11.1 seconds. Results were saved to `outputs/metrics/cross_validation_folds.csv`, `outputs/metrics/cross_validation_summary.csv`, and `outputs/metrics/cross_validation_summary.json`.
- Streamlit launch: not tested because Streamlit is absent from the active runtime.
- Full end-to-end run including SHAP: not run in this stage.

## Fresh hold-out metrics

The following metrics came from the successful `run.py --skip-cv --skip-shap` execution. Threshold = 0.5. They are generated from this code/runtime and should be considered provisional until the environment is installed from a clean environment and the complete pipeline is rerun.

| Model | Accuracy | Precision | Recall/Sensitivity | Specificity | F1 | ROC-AUC | PR-AUC | MCC | Cohen's Kappa | Log Loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7353 | 0.5008 | 0.7888 | 0.7159 | 0.6127 | 0.8420 | 0.6370 | 0.4518 | 0.4264 | 0.4950 |
| Decision Tree | 0.7488 | 0.5181 | 0.7674 | 0.7420 | 0.6185 | 0.8321 | 0.6244 | 0.4605 | 0.4415 | 0.5673 |
| Random Forest | 0.7693 | 0.5476 | 0.7540 | 0.7749 | 0.6344 | 0.8413 | 0.6462 | 0.4850 | 0.4721 | 0.4623 |
| XGBoost | 0.7530 | 0.5233 | 0.7807 | 0.7430 | 0.6266 | 0.8428 | 0.6498 | 0.4729 | 0.4526 | 0.4777 |
| RF + XGBoost hybrid | 0.7644 | 0.5396 | 0.7647 | 0.7643 | 0.6327 | 0.8439 | 0.6510 | 0.4822 | 0.4668 | 0.4665 |

## Five-fold stratified CV results

Cross-validation uses training data only. The hold-out test set was not used for tuning or CV.

| Model | Mean ROC-AUC | SD ROC-AUC | Mean PR-AUC | Mean F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.8449 | 0.0133 | 0.6610 | 0.6259 |
| Decision Tree | 0.8318 | 0.0138 | 0.6176 | 0.6179 |
| Random Forest | 0.8459 | 0.0102 | 0.6590 | 0.6319 |
| XGBoost | 0.8406 | 0.0130 | 0.6511 | 0.6282 |
| RF + XGBoost hybrid | 0.8450 | 0.0116 | 0.6582 | 0.6282 |

## Remaining limitations

- The root-level Streamlit app exists but could not be launched in the active environment because Streamlit is not installed.
- Dependency pins are aligned to versions observed in the runtime, except that Streamlit is a declared but unverified install. A clean virtual-environment installation test is still required.
- This stage does not add the missing SVM and ANN baselines, the additional hybrid candidates, hyperparameter tuning, paired significance tests, or the research paper/literature deliverables. Those are subsequent research stages.
- The successful fast run skips CV and SHAP. CV was separately rerun and saved, but a single fully completed `python run.py` execution with all stages is not yet claimed.
- The results are not yet submission-ready research claims; they must be reproduced in the final chosen environment and reported with the experimental protocol.

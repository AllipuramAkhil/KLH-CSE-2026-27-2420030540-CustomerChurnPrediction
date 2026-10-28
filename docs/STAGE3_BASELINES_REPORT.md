# Stage 3 Report: Baselines and Feature Engineering

## Scope
Added two required individual baselines, SVM and ANN (MLP), to the existing Logistic Regression, Decision Tree, Random Forest, XGBoost, and RF+XGBoost hybrid. Updated plotting palettes so all model names can be rendered. Added a controlled feature-engineering ablation script.

## Execution status
- Python syntax check: all project `.py` files compiled in-process with no syntax errors.
- Fast end-to-end modelling run: completed training and hold-out evaluation for all seven models, saved model artifacts and metrics, and generated model figures after the palette fix. Run command: `python run.py --skip-cv --skip-shap`.
- Five-fold CV: fold-level results and summary for all seven models are present in `outputs/metrics/cross_validation_folds.csv` and `cross_validation_summary.csv`. The full orchestration command was interrupted by the execution environment's 60-second tool limit before the final metadata write; do not treat that particular orchestration as a fully completed run. The CV summary itself contains 7 models × 5 folds × 10 metrics.
- Feature engineering ablation: `python -m src.evaluation.feature_engineering_ablation` completed successfully and wrote `outputs/metrics/feature_engineering_ablation.csv`.
- SHAP and Streamlit launch were not run in this stage.

## Fresh hold-out test results
| Model | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC | PR-AUC | MCC | Kappa | Log Loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7353 | 0.5008 | 0.7888 | 0.7159 | 0.6127 | 0.8420 | 0.6370 | 0.4518 | 0.4264 | 0.4950 |
| Decision Tree | 0.7488 | 0.5181 | 0.7674 | 0.7420 | 0.6185 | 0.8321 | 0.6244 | 0.4605 | 0.4415 | 0.5673 |
| Random Forest | 0.7693 | 0.5476 | 0.7540 | 0.7749 | 0.6344 | 0.8413 | 0.6462 | 0.4850 | 0.4721 | 0.4623 |
| XGBoost | 0.7530 | 0.5233 | 0.7807 | 0.7430 | 0.6266 | 0.8428 | 0.6498 | 0.4729 | 0.4526 | 0.4777 |
| Support Vector Machine (RBF) | 0.7828 | 0.5833 | 0.6364 | 0.8357 | 0.6087 | 0.8249 | 0.6142 | 0.4596 | 0.4588 | 0.4470 |
| Artificial Neural Network (MLP) | 0.7977 | 0.6353 | 0.5588 | 0.8841 | 0.5946 | 0.8365 | 0.6230 | 0.4623 | 0.4606 | 0.4267 |
| Hybrid (RF + XGBoost) | 0.7644 | 0.5396 | 0.7647 | 0.7643 | 0.6327 | 0.8439 | 0.6510 | 0.4822 | 0.4668 | 0.4665 |

## Five-fold stratified CV summary

| Model | Mean ROC-AUC | SD ROC-AUC | Mean PR-AUC | SD PR-AUC | Mean F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8449 | 0.0133 | 0.6610 | 0.0139 | 0.6259 |
| Decision Tree | 0.8318 | 0.0138 | 0.6176 | 0.0306 | 0.6179 |
| Random Forest | 0.8459 | 0.0102 | 0.6590 | 0.0240 | 0.6319 |
| XGBoost | 0.8406 | 0.0130 | 0.6511 | 0.0234 | 0.6282 |
| Support Vector Machine (RBF) | 0.8259 | 0.0110 | 0.6017 | 0.0236 | 0.6093 |
| Artificial Neural Network (MLP) | 0.8396 | 0.0139 | 0.6552 | 0.0251 | 0.5860 |
| Hybrid (RF + XGBoost) | 0.8450 | 0.0116 | 0.6582 | 0.0236 | 0.6282 |

## Feature engineering ablation

| Model | Feature set | Accuracy | F1 | Recall | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|---:|
| Logistic Regression | Original features | 0.7381 | 0.6136 | 0.7834 | 0.8416 | 0.6327 |
| Logistic Regression | Engineered features | 0.7353 | 0.6127 | 0.7888 | 0.8420 | 0.6370 |
| Random Forest | Original features | 0.7700 | 0.6351 | 0.7540 | 0.8414 | 0.6494 |
| Random Forest | Engineered features | 0.7693 | 0.6344 | 0.7540 | 0.8413 | 0.6462 |

### Interpretation of the ablation
The engineered features show a small change rather than a large universal improvement. For Logistic Regression, PR-AUC rises from 0.6327 to 0.6370 and ROC-AUC from 0.8416 to 0.8420, while accuracy decreases from 0.7381 to 0.7353. For Random Forest, the original and engineered feature sets are very similar; the original set is marginally higher on the listed metrics. Therefore, the evidence does not support claiming that feature engineering improves every model. Keep the engineered features only if later cross-validated hybrid comparisons or business interpretability justify their added complexity.

## What is learned
1. ANN has the highest recorded test accuracy (0.7977) but substantially lower recall (0.5588) than Logistic Regression (0.7888), illustrating that accuracy alone is insufficient under class imbalance.
2. Random Forest has the highest mean CV ROC-AUC (0.8459). The hybrid mean CV ROC-AUC is 0.8450, so the current hybrid has not demonstrated a CV ROC-AUC advantage over Random Forest.
3. XGBoost has the highest hold-out PR-AUC in this run (0.6498) among individual baselines, while the hybrid is slightly higher (0.6510).
4. SVM and ANN are now part of the same hold-out evaluation table, but hyperparameters have not yet been systematically tuned.
5. The recorded CV summary is useful for screening candidates, not for claiming statistical significance. The untouched test set should be used only after the model-selection plan is frozen.

## Known limitations
- The full pipeline run was interrupted by the execution environment before final run metadata was written. The successful fast run and completed CV artifacts exist, but a later stage should create one synchronized final run record.
- The Streamlit app has not been launched because Streamlit is not installed in the active runtime.
- SHAP explanations have not been regenerated for the expanded model set.
- No paired significance tests have been run yet.
- No hyperparameter search has been completed.
- No additional hybrid candidate beyond the existing RF/XGBoost weighted soft-voting model has been evaluated yet.
- Model plots now use 300 dpi in `src/visualization/model_plots.py`; EDA plot resolution should also be normalized to 300 dpi in a later pass.

## Files added or updated in this stage
- `src/models/baseline_models.py`: SVM and ANN model definitions.
- `src/config.py`: model names and saved-model paths for SVM and ANN.
- `src/visualization/model_plots.py`: palette entries for the new baselines and 300-dpi model figures.
- `dashboard/app.py`: palette entries for the new baselines.
- `app.py`: root-level Streamlit launcher.
- `src/evaluation/feature_engineering_ablation.py`: original-versus-engineered feature experiment.
- `outputs/metrics/feature_engineering_ablation.csv`: measured ablation results.

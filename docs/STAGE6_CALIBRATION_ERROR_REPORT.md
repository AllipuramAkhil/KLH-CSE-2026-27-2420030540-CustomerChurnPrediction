# Stage 6 Report: Calibration, error analysis and business-facing diagnostics

## Completed
The new module `src/evaluation/calibration_error_analysis.py` was executed successfully on the saved 1,409-row holdout predictions. It produces:

- Probability quality: Brier score, log loss, and 10-bin expected calibration error (ECE).
- Classification diagnostics at the project's fixed 0.50 threshold: precision, recall, F1, MCC, and confusion counts.
- A 300-dpi reliability diagram for the existing RF+XGBoost hybrid, weighted voting, stacking, and Logistic Regression.
- Per-customer prediction/error records for eight models.
- A top-100 list of high-confidence errors for the three leading hybrid candidates.
- Descriptive subgroup diagnostics for Contract, InternetService, tenure group, and MonthlyCharges quartile, retaining segments with at least 30 rows.

Run:
```bash
python -m src.evaluation.calibration_error_analysis
```

## Results observed on the existing holdout

| Model | PR-AUC | Brier (lower is better) | Log loss (lower is better) | ECE (lower is better) | Recall | F1 | False negatives |
|---|---:|---:|---:|---:|---:|---:|---:|
| Weighted soft voting | 0.6573 | 0.1496 | 0.4508 | 0.0964 | 0.7406 | 0.6253 | 97 |
| Existing RF+XGBoost hybrid | 0.6510 | 0.1555 | 0.4665 | 0.1171 | 0.7647 | 0.6327 | 88 |
| Stacking | 0.6464 | 0.1663 | 0.5041 | 0.1566 | 0.7941 | 0.6387 | 77 |
| ANN | 0.6230 | 0.1401 | 0.4267 | 0.0244 | 0.5588 | 0.5946 | 165 |

Interpretation:
- Weighted soft voting has the highest holdout PR-AUC among the listed models and improves Brier score, log loss and ECE relative to the existing RF+XGBoost hybrid in this exploratory comparison.
- Stacking has the highest recall and fewest false negatives among these three hybrid approaches, but its Brier score, log loss and ECE are worse. It may be relevant if missed churners are especially costly, but threshold and probability quality require further validation.
- ANN has the lowest Brier score and ECE in this table, but lower recall and PR-AUC. Calibration quality and ranking/detection quality are different properties; no single metric should determine the winner.
- The subgroup table is descriptive. Segment differences should not be treated as stable fairness findings without uncertainty estimates, predeclared hypotheses, and validation on independent data.

## Output files
- `outputs/metrics/calibration_error_summary.csv`
- `outputs/metrics/per_customer_error_analysis.csv`
- `outputs/metrics/subgroup_error_analysis.csv`
- `outputs/figures/calibration_reliability.png`
- `outputs/reports/high_confidence_error_cases.csv`
- `outputs/reports/calibration_error_analysis_metadata.json`

## Limitations and research integrity
The 1,409-row holdout was already examined in prior model comparisons and is therefore exploratory rather than an untouched confirmatory test set. Do not present these values as independent final validation. The dataset is the IBM Telco Customer Churn sample and is not evidence that a model will achieve the same results in a live business. Before a deployment claim, validate on fresh data, choose a threshold using training/validation data, calibrate if needed, and quantify uncertainty. Customer IDs in the error export are dataset row identifiers; avoid sharing them unnecessarily.

## Next stage
Stage 7 should verify literature and citation details, audit the full pipeline/dashboard, and prepare the reproducible research-paper results only after the remaining experiments have been run. Novelty or superiority claims must be supported by the literature and appropriately designed evaluation.

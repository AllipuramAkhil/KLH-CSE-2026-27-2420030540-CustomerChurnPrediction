# Stage 7 Report: Literature register and reproducibility audit

## 1. Work completed

- Created `docs/VERIFIED_LITERATURE_REGISTER.md` with publisher/official documentation links and a citation checklist.
- Audited Python syntax across the project: 37 Python files parsed successfully.
- Checked expected metrics, reports, and plots for Stages 4–6; the listed files were present.
- Verified that the parameter names in both Stage 5 randomized-search spaces exist on their estimators. This is a static configuration check only; it does not mean the tuning experiment completed.
- Ran the Stage 6 threshold-analysis module successfully. It saved `outputs/metrics/threshold_analysis.csv` and `outputs/figures/threshold_analysis.png`.
- Confirmed the installed runtime has pandas 2.2.3, scikit-learn 1.8.0, XGBoost 3.1.3, SHAP 0.50.0, and Plotly 6.5.2. Streamlit is not installed in this runtime.

## 2. Threshold analysis

The table below uses the already-examined holdout predictions and is exploratory.

| Model | Threshold | Precision | Recall | Specificity | F1 | False negatives |
|---|---:|---:|---:|---:|---:|---:|
| Existing RF+XGBoost hybrid | 0.25 | 0.4322 | 0.9198 | 0.5633 | 0.5880 | 30 |
| Existing RF+XGBoost hybrid | 0.50 | 0.5396 | 0.7647 | 0.7643 | 0.6327 | 88 |
| Existing RF+XGBoost hybrid | 0.65 | 0.6110 | 0.5963 | 0.8628 | 0.6035 | 151 |
| Stacking | 0.25 | 0.4242 | 0.9278 | 0.5449 | 0.5822 | 27 |
| Stacking | 0.50 | 0.5342 | 0.7941 | 0.7498 | 0.6387 | 77 |
| Stacking | 0.65 | 0.5779 | 0.6444 | 0.8300 | 0.6094 | 133 |
| Weighted soft voting | 0.25 | 0.4493 | 0.9011 | 0.6010 | 0.5996 | 37 |
| Weighted soft voting | 0.50 | 0.5410 | 0.7406 | 0.7729 | 0.6253 | 97 |
| Weighted soft voting | 0.65 | 0.6366 | 0.5668 | 0.8831 | 0.5997 | 162 |

Lowering the threshold increases recall and reduces false negatives but also lowers precision and specificity. Raising it generally increases precision/specificity but misses more churners. A threshold must be chosen using business costs and training/validation data, not selected by optimizing the already-inspected test set.

## 3. Dashboard and explainability status

- The dashboard entry points and source files are present (`app.py`, `dashboard/app.py`, `dashboard/components/`).
- Streamlit is specified in `requirements.txt` but is not installed in this execution runtime. The dashboard could not be launched and interactively tested here.
- No SHAP output artifacts were found in the output folder during this audit. The SHAP implementation exists and has fallback logic, but a successful SHAP run is not yet verified.
- The main pipeline supports `--skip-cv` and `--skip-shap`; successful use of those flags should not be described as validation of full CV, SHAP, or Streamlit.
- `requirements.txt` pins packages and recommends Python 3.12. Recreate the environment in a clean virtual environment and run the full pipeline/dashboard before final submission.

## 4. Data and methodological caveats

1. The commonly circulated CSV has 7,043 rows and 21 columns. IBM's current Cognos sample documentation also describes a multi-table sample; cite the exact CSV mirror actually used and record its retrieval date and file hash.
2. The fixed holdout has already been inspected in prior model comparisons. Stage 5 tuning code keeps its parameter search within training CV, but any holdout results from later candidate selection remain exploratory. Prefer a new external/final split or nested cross-validation for confirmatory claims.
3. The subgroup diagnostics are descriptive. Do not claim fairness or causal effects from segment differences alone.
4. The Stage 5 tuning job did not finish in the constrained runtime. No tuned hyperparameters or tuned performance values are reported.
5. The Stage 4 statistical comparisons did not establish a robust significant improvement after multiple-comparison correction. Avoid claims of statistical superiority.
6. Do not claim novelty based only on combining known models and SHAP. The literature register contains recent related work; a fuller literature search and a clearly defined research contribution are needed.

## 5. Reproduction checklist

From the project root, using Python 3.12 in a clean virtual environment:

```bash
python -m pip install -r requirements.txt
python run.py
streamlit run app.py
```

The full pipeline may take a while. Verify that it creates the run metadata, metrics, model files, plots, and SHAP outputs. Then run the hybrid candidate and analysis modules separately as documented in the README. Record command, environment, seed, dataset source/hash, and success/failure in the final report.

## 6. Status summary

| Item | Status |
|---|---|
| Source register created | Complete (initial verified register) |
| Python syntax audit | Passed (37 files) |
| Stage 5 search parameter-name audit | Passed; tuning run still incomplete |
| Threshold analysis | Executed successfully |
| Calibration/error analysis | Executed successfully |
| Streamlit launch test | Not run: dependency absent in audit runtime |
| SHAP execution | Not verified; no generated SHAP files found |
| Final confirmatory evaluation | Pending |

# Stage 8 Report: Research paper draft

## Deliverables
- `docs/STAGE8_RESEARCH_PAPER_DRAFT.docx` — editable academic paper draft.
- `docs/STAGE8_RESEARCH_PAPER_DRAFT.md` — Markdown source for version control and later revision.

## Paper structure
Abstract; Introduction; Related Work; Dataset and Preprocessing; Methods; Results; Discussion; Limitations; Conclusion and Future Work; References; Reproduction Commands.

## Results included
The paper uses saved metrics from the existing model-comparison, cross-validation, hybrid-candidate, paired-significance, calibration, and threshold-analysis CSVs. The local dataset SHA-256 is included in the Methods section for reproducibility.

## Important limitations explicitly disclosed
- The holdout was reused during candidate development, so its results are exploratory.
- Stage 5 hyperparameter tuning is implemented but not completed; no tuned results are claimed.
- SHAP code exists but no generated SHAP artifacts were found in the audit; no SHAP importance rankings are claimed.
- Streamlit is specified in requirements but was not installed in the audit runtime, so dashboard launch remains unverified.
- The current literature register is a starting set of sources, not a systematic review.
- Statistical comparisons do not establish a robust significant winner after the specified correction.

## Before submitting
1. Replace the author/affiliation placeholder.
2. Ask the supervisor whether IEEE formatting and an exact page/word count are required.
3. Complete tuning and verify SHAP/dashboard execution.
4. Consider a fresh final split or nested CV for confirmatory results.
5. Recheck each citation against the publisher PDF and follow the university's required citation style.
6. Add verified figures (model comparison, calibration, threshold) if permitted by the submission format.

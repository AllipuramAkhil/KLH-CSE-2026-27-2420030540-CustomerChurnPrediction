# Verified literature and source register — Stage 7

This register records sources that were located through publisher, conference, or official documentation pages. It is a starting bibliography, not a claim that the literature review is exhaustive. Read each full text and confirm all required metadata before submission.

## References

[1] S. M. Lundberg and S.-I. Lee, “A Unified Approach to Interpreting Model Predictions,” in *Advances in Neural Information Processing Systems 30 (NeurIPS 2017)*, 2017. [Conference paper](https://papers.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html).  
**Use for:** SHAP as a framework for local feature attribution and model explanation.  
**Caution:** SHAP explanations describe model behavior/associations; they do not prove causal effects.

[2] M. Imani, M. Joudaki, A. Beikmohammadi, and H. R. Arabnia, “Customer Churn Prediction: A Systematic Review of Recent Advances, Trends, and Challenges in Machine Learning and Deep Learning,” *Machine Learning and Knowledge Extraction*, vol. 7, no. 3, article 105, 2025. https://doi.org/10.3390/make7030105.  
**Use for:** Broad literature context, prevalence of ensemble methods, imbalance, interpretability, drift, and business-aligned evaluation.  
**Caution:** This is a review, not direct evidence that any particular model in this project is superior.

[3] “Explaining customer churn prediction in telecom industry using tabular machine learning models,” *Machine Learning with Applications*, vol. 17, article 100567, 2024. https://doi.org/10.1016/j.mlwa.2024.100567.  
**Use for:** Telecom churn prediction with local/global explanations and SHAP.  
**Metadata note:** Verify the author list, page/article number, and exact publication metadata from the publisher page or PDF before final IEEE formatting.

[4] IBM, “Telco customer churn,” *IBM Cognos Analytics sample documentation*. [Official sample description](https://www.ibm.com/docs/en/cognos-analytics/12.1.x?topic=samples-telco-customer-churn).  
**Use for:** Description of IBM's Telco churn sample data and the churn field.  
**Important dataset provenance note:** The project uses the commonly circulated 7,043-row, 21-column CSV (`WA_Fn-UseC_-Telco-Customer-Churn.csv`). IBM's current Cognos sample documentation also describes a multi-table sample. Do not imply that every current IBM sample file is identical to the single CSV used here. Record the exact download URL, retrieval date, and file hash in the final project if possible.

[5] scikit-learn developers, “Common pitfalls and recommended practices,” scikit-learn documentation, section on data leakage. [Documentation](https://scikit-learn.org/stable/common_pitfalls.html).  
**Use for:** Keeping preprocessing and feature selection inside training folds and preventing test leakage.

[6] scikit-learn developers, “Probability calibration,” scikit-learn documentation. [Documentation](https://scikit-learn.org/stable/modules/calibration.html).  
**Use for:** Reliability diagrams, Brier score, log loss, and interpretation of calibration metrics. The Brier score is not a pure calibration-only measure; it reflects multiple aspects of probabilistic predictions.

[7] scikit-learn developers, “RandomizedSearchCV,” scikit-learn documentation. [Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html).  
**Use for:** Search protocol and the distinction between sampled parameter search and exhaustive grid search.

[8] “A data-driven approach with explainable artificial intelligence for customer churn prediction in the telecommunications industry,” *Results in Engineering*, vol. 26, article 104629, 2025. https://doi.org/10.1016/j.rineng.2025.104629.  
**Use for:** Recent explainable ensemble work in telecom churn prediction.  
**Caution:** Reported metrics from a different dataset/experimental design are not directly comparable to this project's holdout metrics.

## Evidence and novelty statement

These sources establish that churn prediction, ensemble methods, and SHAP-based explanations are active, well-studied areas. Therefore, do **not** claim that this project is the first to combine churn prediction, hybrid ensembles, and explainable AI. A defensible project contribution can instead be framed as a reproducible comparison of selected hybrid candidates on the chosen public benchmark, with calibration and error analysis, provided the final methodology and metrics are verified.

## Suggested research gap (carefully scoped)

Prior work has investigated both high-performing churn classifiers and post-hoc explainability. A reasonable project-level question is whether a compact, reproducible comparison of diverse hybrid candidates—evaluated with PR-AUC, recall, calibration, paired uncertainty estimates, and segment-level error diagnostics—provides a more complete decision-oriented assessment than accuracy-only model ranking on this benchmark. This is a **motivation to test**, not an established novel gap until a fuller systematic search confirms it.

## Citation checklist before final submission

- [ ] Download/read the full text of each paper used in the final literature survey.
- [ ] Verify author names, title, journal/conference, volume, issue, pages/article number, year, and DOI.
- [ ] Include citations only for statements the source actually supports.
- [ ] Do not compare headline accuracy values across different datasets or split protocols as if they were directly comparable.
- [ ] Clearly separate prior literature findings from this project's experimental results.
- [ ] Record the exact CSV source URL, retrieval date, file hash, package versions, and random seed.

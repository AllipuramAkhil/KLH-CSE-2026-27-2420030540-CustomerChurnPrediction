# Likely viva questions with student-level answers

Project numbers quoted below are from the executed run (test set of 1,409 customers, seed 42).

**1. Why did you choose customer churn prediction?**
Because it is a real business problem with clear value: keeping an existing customer costs less than winning a new one, and the data (services, billing, tenure) is structured and public, so I can build a complete analytics pipeline and compare methods fairly.

**2. Why is churn prediction a classification problem?**
The outcome is a category - the customer either churns (1) or stays (0). The model predicts a class and also the probability of belonging to the churn class.

**3. Why did you select Logistic Regression as a baseline?**
It is the standard simple benchmark: fast, interpretable (coefficients show direction of influence) and hard to overfit. If a complex model cannot beat it clearly, the complexity is not justified. In this project Logistic Regression was surprisingly strong (test ROC-AUC 0.842).

**4. Why did you select Decision Tree?**
It gives human-readable if-then rules and captures non-linear relationships, so it is a natural step up from a linear model. I limited its depth (5) and leaf size (20) so it does not memorise the training data. It was the weakest model here, which shows why ensembles are used.

**5. Why Random Forest?**
It trains many decorrelated trees on bootstrap samples and averages them (bagging). This reduces the variance of a single tree, handles non-linear effects and is robust to noisy features.

**6. Why XGBoost?**
It is a boosting method: trees are built one after another, each correcting the errors of the previous ones, with regularisation. It is very strong on structured/tabular data and captures complex interactions.

**7. Why combine Random Forest and XGBoost?**
They learn in different ways (bagging vs boosting), so their errors need not be identical; averaging their probabilities can give more stable predictions. I treated this as a hypothesis to test, not a guarantee.

**8. What makes the proposed approach hybrid?**
It combines two different ensemble learning strategies into one predictor: the final churn probability is a weighted average of the Random Forest and XGBoost probabilities, implemented in `WeightedSoftVotingClassifier`.

**9. What is soft voting?**
Each model outputs a probability for each class; soft voting averages these probabilities (optionally with weights) and picks the class with the higher average. Hard voting would only count each model's final class vote. Here: `P_hybrid = 0.5*P_RF + 0.5*P_XGB`, churn if `P_hybrid >= 0.5`.

**10. What is the difference between bagging and boosting?**
Bagging trains models independently on random bootstrap samples and averages them - it mainly reduces variance (Random Forest). Boosting trains models sequentially, each focusing on the mistakes of the previous ones - it mainly reduces bias (XGBoost).

**11. Why is accuracy insufficient for churn prediction?**
Only about 26.5 % of customers churn, so a model that says "nobody churns" would be 73.5 % accurate yet find no churners. We need recall, precision, specificity, PR-AUC, MCC and others to see how well the minority class is handled.

**12. What is ROC-AUC?**
The ROC curve plots true-positive rate against false-positive rate over all possible thresholds; AUC is the area under it. It equals the probability that a random churner receives a higher score than a random non-churner: 0.5 is guessing, 1.0 is perfect. Our models score about 0.83-0.84.

**13. What is PR-AUC?**
The area under the precision-recall curve (average precision). It focuses on the positive (churn) class and is more informative than ROC-AUC when the positive class is rare. The no-skill baseline equals the churn rate (~0.27); our models reach about 0.62-0.66.

**14. What is specificity?**
`TN / (TN + FP)`: of the customers who really do not churn, the share the model correctly identifies as staying. Low specificity means many unnecessary retention offers.

**15. What is sensitivity?**
`TP / (TP + FN)` - identical to recall: of the customers who really churn, the share the model catches. Low sensitivity means lost customers we did not warn about. For the hybrid: 288 of 374 churners caught (77.0 %).

**16. What is MCC?**
Matthews Correlation Coefficient: a correlation between predicted and actual classes computed from all four cells of the confusion matrix (TP, TN, FP, FN). It ranges from -1 to +1 and stays honest on imbalanced data because a model that ignores a class cannot score high.

**17. What is Cohen's Kappa?**
It measures agreement between predictions and truth *beyond what chance alone would give*. 0 means chance-level agreement, 1 means perfect agreement.

**18. Why use Log Loss?**
Accuracy and F1 only look at the final class. Log Loss evaluates the predicted *probabilities*: confident wrong predictions are punished heavily, so it shows how good the probabilities are. Lower is better. In this project it is inflated by class weighting, so I do not treat the probabilities as calibrated risks.

**19. What is data leakage?**
When information that would not be available at prediction time - for example test-set statistics or the target itself - influences training. It makes results look better than they really are.

**20. How did you prevent data leakage?**
I split into train/test first; imputation, scaling and one-hot encoding live inside an sklearn Pipeline, so they are fitted only on training data (and re-fitted inside each CV fold); engineered features use only the same row; `customerID` is dropped; the test set was used once for the final evaluation; model selection and weight checks used cross-validation on the training data only.

**21. Why use stratified splitting?**
Churners are a minority. Stratification keeps the same churn ratio (26.5 %) in the training set, test set and every CV fold, so evaluations are stable and comparable.

**22. Why use cross-validation?**
One split can be lucky or unlucky. 5-fold CV trains and validates on five different splits of the training data and gives a mean and a standard deviation, showing how stable each model is - and lets me compare models without touching the test set.

**23. What does the confusion matrix tell us?**
It counts correct and wrong predictions by type: TN (stayers correctly kept), FP (stayers wrongly flagged), FN (churners missed), TP (churners caught). All threshold-based metrics are derived from it. Hybrid on the test set: TN 793, FP 242, FN 86, TP 288.

**24. What are the limitations of the project?**
One dataset and one random split; a static snapshot without time; class weighting makes probabilities uncalibrated and the 0.5 threshold is not cost-optimised; the hybrid's improvement over its own components is small and not statistically significant; SHAP is approximate and shows association, not causation.

**25. How would you improve the system in the future?**
Calibrate the probabilities; choose the threshold from the real cost of a missed churner vs. a retention offer; tune hyper-parameters with nested CV; try stacking with a meta-learner; validate on time-ordered data; check fairness across customer groups; monitor drift; deploy as an API.

---
### Extra questions you may get
**Did your hybrid beat the baselines?** Partly. It has the best ROC-AUC, F1, MCC and Kappa on the test set, but not the best PR-AUC, Log Loss, Recall, Precision, Specificity or Accuracy, and the differences to Random Forest, XGBoost and Logistic Regression are within statistical noise (bootstrap CIs include 0). It is clearly better than a single Decision Tree. Say this openly - it is the honest scientific result.

**Why is the hybrid's gain so small?** Random Forest and XGBoost produce very similar probabilities here (correlation about 0.98, different class on only about 6 % of customers), so averaging has little to correct.

**Why is a simple model so strong?** Much of the churn signal (contract, tenure, internet type, payment method) is strong and roughly additive, so a regularised linear model captures most of it. This was not tested separately.

**Does SHAP show what causes churn?** No. It shows what the model associates with higher or lower predicted churn risk; causal claims would need experiments.

# Hybrid Candidate Architectures

## Candidate A: Stacking ensemble (RF + XGBoost + SVM, Logistic Regression meta-learner)

```mermaid
flowchart TD
    A[Prescribed Telco customer data] --> B[Leakage-safe stratified train/test split]
    B --> C[Training fold only]
    C --> D[Row-wise feature engineering]
    D --> E[Fold-fitted imputation, scaling and one-hot encoding]
    E --> F1[Random Forest]
    E --> F2[XGBoost]
    E --> F3[RBF SVM]
    F1 --> G[Out-of-fold class probabilities]
    F2 --> G
    F3 --> G
    G --> H[Logistic Regression meta-learner]
    H --> I[Churn probability]
    I --> J[Threshold and evaluation]
    B --> K[Held-out test set kept untouched]
    J --> K
```

The stacking meta-learner is trained from out-of-fold predictions, not from base-model predictions on the same rows used to fit those base models. This reduces meta-level leakage.

## Candidate B: Weighted soft voting

```mermaid
flowchart TD
    A[Fold-fitted feature pipeline] --> B[Random Forest probability]
    A --> C[XGBoost probability]
    A --> D[RBF SVM probability]
    B --> E[Weighted probability average]
    C --> E
    D --> E
    E --> F[Churn probability]
    F --> G[Thresholded class and ranking metrics]
```

The current candidate weights are RF = 0.2, XGBoost = 0.6 and SVM = 0.2. They are a preselected experimental configuration, not yet tuned weights.

## Candidate C: RF feature selection + ANN

```mermaid
flowchart TD
    A[Fold training data] --> B[Preprocessing]
    B --> C[Random Forest feature importance]
    C --> D[Select features above median importance]
    D --> E[ANN / MLP]
    E --> F[Churn probability]
```

## Candidate D: XGBoost leaf representation + ANN

```mermaid
flowchart TD
    A[Fold training data] --> B[Preprocessing]
    B --> C[Fit XGBoost trees]
    C --> D[Terminal leaf indices per tree]
    D --> E[One-hot encode leaf indices]
    E --> F[ANN / MLP]
    F --> G[Churn probability]
```

## Candidate E: Chi-square feature selection + ensemble

```mermaid
flowchart TD
    A[Fold training data] --> B[Preprocessing]
    B --> C[Min-max transform to non-negative values]
    C --> D[Chi-square feature ranking]
    D --> E[Select top 25 encoded features]
    E --> F[RF + XGBoost soft voting]
    F --> G[Churn probability]
```

All candidates use the prescribed CSV, a common fixed hold-out split, and five-fold stratified CV on the training portion. Preprocessing and selection are fitted inside the pipeline on each training fold.

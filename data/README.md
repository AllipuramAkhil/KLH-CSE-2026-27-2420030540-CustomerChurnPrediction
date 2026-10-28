# Data

## Dataset
**IBM Telco Customer Churn** sample dataset - 7,043 customers, 21 columns (19 input features, `customerID`, and the target `Churn`).
Each row is a telecom customer: demographics, subscribed services, account/billing information and whether the customer left within the last month.

Sources (both host the same file):
- IBM sample data: <https://github.com/IBM/telco-customer-churn-on-icp4d> (file `data/Telco-Customer-Churn.csv`)
- Kaggle mirror: <https://www.kaggle.com/datasets/blastchar/telco-customer-churn> (file `WA_Fn-UseC_-Telco-Customer-Churn.csv`)

The data is a fictional sample published by IBM for analytics demonstrations. Check the source repository's licence if you plan to redistribute it outside your course.

## Folders
| Folder | Content |
|---|---|
| `raw/` | The original CSV, **never modified**: `WA_Fn-UseC_-Telco-Customer-Churn.csv` |
| `processed/` | Written by `python run.py`: `telco_clean.csv`, `train.csv`, `test.csv` (git-ignored, safe to delete) |

## If the file is missing
`python run.py` tries to download it automatically from the IBM repository. If you are offline, download the CSV from one of the links above and save it as
`data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` (any `.csv` in `data/raw/` with the Telco columns is also detected).

## Target
`Churn`: `Yes` -> 1 (customer churned, the *positive class*), `No` -> 0.

## Known data quirk
`TotalCharges` is stored as text; 11 rows (all with `tenure = 0`) contain a blank string. They are converted to numeric and set to 0.0 because a customer with zero months of tenure has not been billed yet.

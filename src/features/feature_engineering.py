"""Churn-specific feature engineering.

`ChurnFeatureEngineer` is a *stateless* sklearn transformer: every new column is
computed row by row from the same row's raw values (no dataset-wide statistic is
used), so applying it before or inside the pipeline cannot leak information
from the test set.  It is placed inside the Pipeline so that the saved model
accepts raw customer records.

Engineered features (all documented in docs/methodology.md)
-----------------------------------------------------------
tenure_group             categorical  Tenure bucket: 0-12, 13-24, 25-48, 49-60, 61+ months.
                                      Churn risk is strongly non-linear in tenure.
service_count            numeric      Number of subscribed services (phone, multiple lines,
                                      internet, security, backup, protection, tech support,
                                      streaming TV, streaming movies).  Measures engagement.
protection_service_count numeric      Number of security/support add-ons (OnlineSecurity,
                                      OnlineBackup, DeviceProtection, TechSupport).
streaming_service_count  numeric      StreamingTV + StreamingMovies.
avg_monthly_charge       numeric      TotalCharges / tenure (historical average monthly bill;
                                      falls back to MonthlyCharges when tenure == 0).
charge_delta             numeric      MonthlyCharges - avg_monthly_charge: how far the current
                                      bill is above/below the customer's historical average.
charges_per_service      numeric      MonthlyCharges / max(service_count, 1): price paid per service.
is_month_to_month        binary       Contract == Month-to-month (no commitment).
auto_payment             binary       Payment by automatic bank transfer or credit card.
is_electronic_check      binary       PaymentMethod == Electronic check.
has_family               binary       Has a partner or dependents.
mtm_fiber                binary       Interaction: month-to-month contract AND fiber-optic internet
                                      (a segment observed in EDA to have a high churn rate).
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

TENURE_BINS = [-1, 12, 24, 48, 60, np.inf]
TENURE_LABELS = ["0-12", "13-24", "25-48", "49-60", "61+"]

ADDON_SERVICES = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
                  "StreamingTV", "StreamingMovies"]
PROTECTION_SERVICES = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
STREAMING_SERVICES = ["StreamingTV", "StreamingMovies"]
AUTO_PAYMENT_METHODS = ["Bank transfer (automatic)", "Credit card (automatic)"]

ENGINEERED_NUMERIC = [
    "service_count", "protection_service_count", "streaming_service_count",
    "avg_monthly_charge", "charge_delta", "charges_per_service",
]
ENGINEERED_BINARY = ["is_month_to_month", "auto_payment", "is_electronic_check",
                     "has_family", "mtm_fiber"]
ENGINEERED_CATEGORICAL = ["tenure_group"]
ENGINEERED_FEATURES = ENGINEERED_NUMERIC + ENGINEERED_BINARY + ENGINEERED_CATEGORICAL


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """Append the engineered churn features to the raw customer columns."""

    def fit(self, X, y=None):
        self._check_frame(X)
        self.output_columns_ = list(self.transform(X.head(2)).columns)
        return self

    @staticmethod
    def _check_frame(X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ChurnFeatureEngineer expects a pandas DataFrame with raw customer columns.")

    def transform(self, X):
        self._check_frame(X)
        df = X.copy()

        df["tenure_group"] = pd.cut(df["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS).astype(str)

        has_phone = (df["PhoneService"] == "Yes").astype(int)
        multi_lines = (df["MultipleLines"] == "Yes").astype(int)
        has_internet = (df["InternetService"] != "No").astype(int)
        addons = sum((df[c] == "Yes").astype(int) for c in ADDON_SERVICES)
        df["service_count"] = has_phone + multi_lines + has_internet + addons
        df["protection_service_count"] = sum((df[c] == "Yes").astype(int) for c in PROTECTION_SERVICES)
        df["streaming_service_count"] = sum((df[c] == "Yes").astype(int) for c in STREAMING_SERVICES)

        # Historical average bill; NaN TotalCharges stays NaN (imputed later in the pipeline)
        safe_tenure = df["tenure"].replace(0, np.nan)
        avg_charge = df["TotalCharges"] / safe_tenure
        df["avg_monthly_charge"] = avg_charge.where(df["tenure"] > 0, df["MonthlyCharges"])
        df["charge_delta"] = df["MonthlyCharges"] - df["avg_monthly_charge"]
        df["charges_per_service"] = df["MonthlyCharges"] / df["service_count"].clip(lower=1)

        df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
        df["auto_payment"] = df["PaymentMethod"].isin(AUTO_PAYMENT_METHODS).astype(int)
        df["is_electronic_check"] = (df["PaymentMethod"] == "Electronic check").astype(int)
        df["has_family"] = ((df["Partner"] == "Yes") | (df["Dependents"] == "Yes")).astype(int)
        df["mtm_fiber"] = ((df["Contract"] == "Month-to-month")
                           & (df["InternetService"] == "Fiber optic")).astype(int)
        return df

    def get_feature_names_out(self, input_features=None):
        return np.array(self.output_columns_, dtype=object)

"""Streamlit dashboard:  streamlit run dashboard/app.py

Reads the models and results saved by `python run.py`; nothing is retrained here.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

from dashboard.components import data_access as da  # noqa: E402
from dashboard.components import prediction as pr  # noqa: E402
from src import config  # noqa: E402
from src.eda.exploratory_analysis import churn_rate_by, dataset_overview  # noqa: E402
from src.features.feature_engineering import TENURE_LABELS, ChurnFeatureEngineer  # noqa: E402

st.set_page_config(page_title="Customer Churn Prediction", page_icon=":bar_chart:", layout="wide")

CHURN_COLORS = {"Not churned": "#4C78A8", "Churned": "#E45756"}
MODEL_COLORS = {"Logistic Regression": "#4C78A8", "Decision Tree": "#F58518", "Random Forest": "#54A24B",
                "XGBoost": "#B279A2", "Support Vector Machine (RBF)": "#72B7B2",
                "Artificial Neural Network (MLP)": "#FF9DA6", config.HYBRID_NAME: "#E45756"}
PAGES = ["1. Overview", "2. EDA", "3. Model Comparison", "4. Model Evaluation",
         "5. Churn Prediction", "6. Explainability"]


# --------------------------------------------------------------------------- helpers
def need_data():
    df = da.load_clean_data()
    if df is None:
        st.error("Dataset not found. Place the IBM Telco CSV in data/raw/ (see data/README.md) or run `python run.py`.")
        st.stop()
    return df


def need(obj, message=da.NOT_RUN_MESSAGE):
    if obj is None:
        st.warning(message)
        st.stop()
    return obj


def label_churn(df):
    out = df.copy()
    out["Churn status"] = out[config.TARGET_COLUMN].map({0: "Not churned", 1: "Churned"})
    return out


def rate_bar(df, column, title, order=None):
    rates = churn_rate_by(df, column)
    if order:
        rates[column] = pd.Categorical(rates[column], categories=order, ordered=True)
        rates = rates.sort_values(column)
    fig = px.bar(rates, x=column, y="churn_rate_pct", text=rates["churn_rate_pct"].round(1).astype(str) + "%",
                 hover_data={"customers": True, "churned": True}, title=title,
                 labels={"churn_rate_pct": "Churn rate (%)"}, color_discrete_sequence=["#E45756"])
    fig.add_hline(y=100 * df[config.TARGET_COLUMN].mean(), line_dash="dash", line_color="gray",
                  annotation_text="overall churn rate")
    fig.update_yaxes(rangemode="tozero")
    return fig


# --------------------------------------------------------------------------- pages
def page_overview():
    st.header("Overview")
    df = need_data()
    info = dataset_overview(df)
    c = st.columns(5)
    c[0].metric("Customers", f"{info['n_rows']:,}")
    c[1].metric("Input features", info["n_features"])
    c[2].metric("Churn rate", f"{100 * info['churn_rate']:.2f}%")
    c[3].metric("Non-churn rate", f"{100 * info['non_churn_rate']:.2f}%")
    c[4].metric("Churned customers", f"{info['n_churn']:,}")
    st.caption("Source: IBM Telco Customer Churn sample dataset (7,043 customers).")

    left, mid, right = st.columns(3)
    counts = label_churn(df)["Churn status"].value_counts().reset_index()
    counts.columns = ["Churn status", "Customers"]
    left.plotly_chart(px.pie(counts, names="Churn status", values="Customers", hole=0.45, title="Churn distribution",
                             color="Churn status", color_discrete_map=CHURN_COLORS), width="stretch")
    contract = df["Contract"].value_counts().reset_index()
    contract.columns = ["Contract", "Customers"]
    mid.plotly_chart(px.bar(contract, x="Contract", y="Customers", title="Contract distribution",
                            color_discrete_sequence=["#4C78A8"]), width="stretch")
    right.plotly_chart(px.histogram(df, x="tenure", nbins=36, title="Tenure distribution (months)",
                                    color_discrete_sequence=["#54A24B"]), width="stretch")
    meta = da.read_json(str(config.REPORTS_DIR / "run_metadata.json"))
    if meta:
        with st.expander("Reproducibility details of the last training run"):
            st.json(meta)


def page_eda():
    st.header("Exploratory data analysis")
    df = label_churn(need_data())
    tabs = st.tabs(["Contract", "Payment method", "Internet service", "Monthly charges", "Tenure", "Senior citizen"])
    with tabs[0]:
        st.plotly_chart(rate_bar(df, "Contract", "Churn rate by contract type"), width="stretch")
    with tabs[1]:
        st.plotly_chart(rate_bar(df, "PaymentMethod", "Churn rate by payment method"), width="stretch")
    with tabs[2]:
        st.plotly_chart(rate_bar(df, "InternetService", "Churn rate by internet service"), width="stretch")
    with tabs[3]:
        kind = st.radio("Chart type", ["Box plot", "Histogram"], horizontal=True, key="mc_kind")
        if kind == "Box plot":
            fig = px.box(df, x="Churn status", y="MonthlyCharges", color="Churn status",
                         color_discrete_map=CHURN_COLORS, title="Monthly charges by churn status")
        else:
            fig = px.histogram(df, x="MonthlyCharges", color="Churn status", barmode="overlay", opacity=0.65,
                               nbins=40, color_discrete_map=CHURN_COLORS, title="Monthly charges by churn status")
        st.plotly_chart(fig, width="stretch")
    with tabs[4]:
        fe = ChurnFeatureEngineer().transform(df[config.RAW_FEATURE_COLUMNS])
        fe[config.TARGET_COLUMN] = df[config.TARGET_COLUMN].values
        a, b = st.columns(2)
        a.plotly_chart(px.histogram(df, x="tenure", color="Churn status", barmode="overlay", opacity=0.65, nbins=36,
                                    color_discrete_map=CHURN_COLORS, title="Tenure by churn status"),
                       width="stretch")
        b.plotly_chart(rate_bar(fe, "tenure_group", "Churn rate by tenure group (months)", TENURE_LABELS),
                       width="stretch")
    with tabs[5]:
        senior = df.assign(SeniorCitizen=df["SeniorCitizen"].map({0: "Not senior", 1: "Senior"}))
        st.plotly_chart(rate_bar(senior, "SeniorCitizen", "Churn rate: senior citizens vs others"),
                        width="stretch")
    st.caption("Churn rates are descriptive associations, not proof of cause.")


def page_comparison():
    st.header("Model comparison (held-out test set)")
    comp = need(da.read_csv("model_comparison.csv"))
    st.info("The hybrid (Random Forest + XGBoost soft voting) is the *proposed* model and is highlighted in red. "
            "Whether it is better than the baselines is decided by the numbers below, metric by metric.")
    lower_better = {"Log Loss"}
    styled = comp.style.format({c: "{:.4f}" for c in config.METRIC_COLUMNS}).apply(
        lambda row: ["background-color: #fde0dd" if row["Model"] == config.HYBRID_NAME else "" for _ in row], axis=1)
    st.dataframe(styled, width="stretch", hide_index=True)

    best = []
    for m in config.METRIC_COLUMNS:
        row = comp.sort_values(m, ascending=m in lower_better).iloc[0]
        best.append({"Metric": m, "Best model": row["Model"], "Value": round(row[m], 4),
                     "Hybrid is best": "Yes" if row["Model"] == config.HYBRID_NAME else "No"})
    st.subheader("Best model per metric")
    st.dataframe(pd.DataFrame(best), width="stretch", hide_index=True)

    metrics = st.multiselect("Metrics to plot", config.METRIC_COLUMNS,
                             default=["ROC-AUC", "PR-AUC", "F1", "MCC", "Sensitivity", "Specificity", "Log Loss"])
    if metrics:
        long = comp.melt(id_vars="Model", value_vars=metrics, var_name="Metric", value_name="Value")
        fig = px.bar(long, x="Metric", y="Value", color="Model", barmode="group", color_discrete_map=MODEL_COLORS,
                     text=long["Value"].round(3))
        fig.update_yaxes(rangemode="tozero")
        st.plotly_chart(fig, width="stretch")
        st.caption("Bars start at zero. Log Loss: lower is better; all other metrics: higher is better.")

    cv = da.read_csv("cross_validation_summary.csv")
    if cv is not None:
        st.subheader("5-fold cross-validation on the training data (mean and standard deviation)")
        cols = ["Model"] + [f"{m} {s}" for m in ("ROC-AUC", "PR-AUC", "F1", "MCC") for s in ("Mean", "Std")]
        st.dataframe(cv[cols].style.format({c: "{:.4f}" for c in cols[1:]}), width="stretch", hide_index=True)
    boot = da.read_csv("bootstrap_hybrid_vs_baselines.csv")
    if boot is not None:
        with st.expander("Is the hybrid's difference statistically meaningful? (paired bootstrap on the test set)"):
            st.dataframe(boot, width="stretch", hide_index=True)
            st.caption("If the 95% interval (ci_low to ci_high) contains 0, the difference could be due to chance.")


def page_evaluation():
    st.header("Model evaluation")
    comp = need(da.read_csv("model_comparison.csv"))
    preds = need(da.load_predictions())
    confusion = need(da.read_json(str(config.METRICS_DIR / "confusion_matrices.json")))
    names = list(config.MODEL_NAMES.values())
    name = st.selectbox("Model", names, index=names.index(config.HYBRID_NAME))
    key = next(k for k, v in config.MODEL_NAMES.items() if v == name)

    counts = confusion[name]
    cm = [[counts["TN"], counts["FP"]], [counts["FN"], counts["TP"]]]
    left, right = st.columns(2)
    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", x=["Predicted: No churn", "Predicted: Churn"],
                       y=["Actual: No churn", "Actual: Churn"], title=f"Confusion matrix - {name}")
    fig_cm.update_coloraxes(showscale=False)
    left.plotly_chart(fig_cm, width="stretch")
    row = comp[comp["Model"] == name].iloc[0]
    right.subheader("Classification metrics")
    right.dataframe(pd.DataFrame({"Metric": config.METRIC_COLUMNS, "Value": [round(row[m], 4) for m in config.METRIC_COLUMNS]}),
                    hide_index=True, width="stretch")
    right.caption("Positive class = churn. Sensitivity/Recall = share of real churners caught; "
                  "Specificity = share of real non-churners correctly identified.")

    a, b = st.columns(2)
    fpr, tpr, _ = roc_curve(preds["y_true"], preds[f"prob_{key}"])
    roc = go.Figure(go.Scatter(x=fpr, y=tpr, mode="lines", name=name, line=dict(color=MODEL_COLORS[name], width=3)))
    roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash", color="gray")))
    roc.update_layout(title=f"ROC curve (AUC = {row['ROC-AUC']:.3f})", xaxis_title="False positive rate",
                      yaxis_title="True positive rate")
    a.plotly_chart(roc, width="stretch")
    prec, rec, _ = precision_recall_curve(preds["y_true"], preds[f"prob_{key}"])
    pr_fig = go.Figure(go.Scatter(x=rec, y=prec, mode="lines", name=name, line=dict(color=MODEL_COLORS[name], width=3)))
    pr_fig.add_hline(y=preds["y_true"].mean(), line_dash="dash", line_color="gray", annotation_text="no-skill")
    pr_fig.update_layout(title=f"Precision-Recall curve (AP = {row['PR-AUC']:.3f})", xaxis_title="Recall",
                         yaxis_title="Precision", yaxis_range=[0, 1.02])
    b.plotly_chart(pr_fig, width="stretch")

    with st.expander("All models: ROC and PR curves together"):
        img1, img2 = da.figure_path("07_roc_curves.png"), da.figure_path("08_precision_recall_curves.png")
        c1, c2 = st.columns(2)
        if img1:
            c1.image(str(img1))
        if img2:
            c2.image(str(img2))


def single_prediction_form(pipeline):
    st.subheader("Single customer")
    c1, c2, c3 = st.columns(3)
    with c1:
        gender = st.selectbox("Gender", pr.CATEGORY_OPTIONS["gender"])
        senior = st.selectbox("Senior citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", pr.CATEGORY_OPTIONS["Partner"], index=1)
        dependents = st.selectbox("Dependents", pr.CATEGORY_OPTIONS["Dependents"], index=1)
        tenure = st.number_input("Tenure (months)", 0, 72, 12)
        phone = st.selectbox("Phone service", pr.CATEGORY_OPTIONS["PhoneService"])
        multi = st.selectbox("Multiple lines", ["No phone service"] if phone == "No" else ["No", "Yes"])
    with c2:
        internet = st.selectbox("Internet service", pr.CATEGORY_OPTIONS["InternetService"], index=1)
        addon_options = ["No internet service"] if internet == "No" else ["No", "Yes"]
        security = st.selectbox("Online security", addon_options)
        backup = st.selectbox("Online backup", addon_options)
        protection = st.selectbox("Device protection", addon_options)
        support = st.selectbox("Tech support", addon_options)
        tv = st.selectbox("Streaming TV", addon_options)
        movies = st.selectbox("Streaming movies", addon_options)
    with c3:
        contract = st.selectbox("Contract", pr.CATEGORY_OPTIONS["Contract"])
        paperless = st.selectbox("Paperless billing", pr.CATEGORY_OPTIONS["PaperlessBilling"])
        payment = st.selectbox("Payment method", pr.CATEGORY_OPTIONS["PaymentMethod"])
        monthly = st.number_input("Monthly charges", 0.0, 200.0, 70.0, step=0.5)
        total = st.number_input("Total charges", 0.0, 10000.0, float(round(monthly * tenure, 2)), step=1.0)
        st.caption("Tip: total charges is roughly monthly charges x tenure.")

    if st.button("Predict churn", type="primary"):
        row = pd.DataFrame([{
            "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0, "Partner": partner,
            "Dependents": dependents, "tenure": tenure, "PhoneService": phone, "MultipleLines": multi,
            "InternetService": internet, "OnlineSecurity": security, "OnlineBackup": backup,
            "DeviceProtection": protection, "TechSupport": support, "StreamingTV": tv, "StreamingMovies": movies,
            "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
            "MonthlyCharges": monthly, "TotalCharges": total}])
        result = pr.predict_frame(pipeline, row).iloc[0]
        prob = float(result["churn_probability"])
        left, right = st.columns([1, 2])
        left.metric("Churn probability", f"{100 * prob:.1f}%")
        if result["churn_prediction"] == 1:
            left.error(f"Prediction: CHURN (probability >= {config.DECISION_THRESHOLD})")
        else:
            left.success(f"Prediction: NO CHURN (probability < {config.DECISION_THRESHOLD})")
        gauge = go.Figure(go.Indicator(mode="gauge+number", value=100 * prob, number={"suffix": "%"},
                                       gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#E45756"},
                                              "threshold": {"line": {"color": "black", "width": 3},
                                                            "value": 100 * config.DECISION_THRESHOLD}}))
        gauge.update_layout(height=260, margin=dict(t=30, b=10))
        right.plotly_chart(gauge, width="stretch")
        st.caption("Estimated by the hybrid (RF + XGBoost) model. It reflects statistical patterns in the training "
                   "data and is a risk score, not a certainty. Probabilities are inflated by class weighting.")


def batch_prediction(pipeline):
    st.subheader("Batch prediction (CSV upload)")
    st.write("Required columns: " + ", ".join(config.RAW_FEATURE_COLUMNS) + ". Optional: `customerID`.")
    df_ref = da.load_clean_data()
    if df_ref is not None:
        template = df_ref.head(5).drop(columns=[config.TARGET_COLUMN])
        st.download_button("Download a sample template", template.to_csv(index=False).encode("utf-8"),
                           "churn_batch_template.csv", "text/csv")
    upload = st.file_uploader("Upload a CSV file", type="csv")
    if upload is None:
        return
    try:
        raw = pd.read_csv(upload)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read the file as CSV: {exc}")
        return
    raw.columns = [str(c).strip() for c in raw.columns]
    missing = [c for c in config.RAW_FEATURE_COLUMNS if c not in raw.columns]
    if missing:
        st.error("Validation failed - missing required column(s): " + ", ".join(missing))
        return
    unknown = pr.unknown_categories(raw)
    if unknown:
        st.warning("Unrecognised category values (the model ignores them): "
                   + "; ".join(f"{k}: {v}" for k, v in unknown.items()))
    try:
        result = pr.predict_frame(pipeline, raw)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Prediction failed: {exc}")
        return
    id_col = pr.find_id_column(raw)
    out = result.copy()
    if id_col:
        out.insert(0, "customer_id", raw[id_col].values)
    st.success(f"Scored {len(out):,} customers; {int(out['churn_prediction'].sum()):,} flagged as likely churn.")
    st.dataframe(out.round({"churn_probability": 4}), width="stretch", hide_index=True)
    st.download_button("Download predictions (CSV)", out.to_csv(index=False).encode("utf-8"),
                       "churn_predictions.csv", "text/csv")


def page_prediction():
    st.header("Churn prediction (hybrid RF + XGBoost model)")
    pipeline, error = da.load_hybrid_pipeline()
    if error:
        st.warning(error)
        st.stop()
    tab_single, tab_batch = st.tabs(["Single customer", "Batch (CSV)"])
    with tab_single:
        single_prediction_form(pipeline)
    with tab_batch:
        batch_prediction(pipeline)


def page_explainability():
    st.header("Explainability")
    st.info("These charts show which inputs the model associates with higher or lower predicted churn risk. "
            "They describe the model, not causes of churn.")
    perm = da.read_csv("feature_importance_permutation.csv")
    shap_df = da.read_csv("shap_importance.csv")
    tree = da.read_csv("feature_importance_tree_based.csv")
    if perm is None and shap_df is None and tree is None:
        st.warning(da.NOT_RUN_MESSAGE)
        st.stop()
    if perm is not None:
        top = perm.head(12).iloc[::-1]
        fig = px.bar(top, x="importance_mean", y="feature", error_x="importance_std", orientation="h",
                     title="Permutation importance of the hybrid (drop in test ROC-AUC when a column is shuffled)",
                     color_discrete_sequence=["#4C78A8"])
        st.plotly_chart(fig, width="stretch")
    if shap_df is not None:
        label = shap_df["explained_model"].iloc[0] if "explained_model" in shap_df else "hybrid"
        top = shap_df.head(15).iloc[::-1]
        st.plotly_chart(px.bar(top, x="mean_abs_shap", y="feature", orientation="h",
                               title=f"Mean |SHAP value| - {label}", color_discrete_sequence=["#E45756"]),
                        width="stretch")
        img = da.figure_path("13_shap_summary_beeswarm.png")
        if img:
            st.image(str(img), caption="SHAP summary plot: each dot is a customer; right = higher predicted churn risk.")
    else:
        st.caption("SHAP output not available - showing feature importance instead.")
    if tree is not None:
        with st.expander("Tree-based importance (Random Forest + XGBoost, averaged with the hybrid weights)"):
            top = tree.head(15).iloc[::-1]
            st.plotly_chart(px.bar(top, x="hybrid_importance", y="feature", orientation="h",
                                   color_discrete_sequence=["#54A24B"]), width="stretch")


# --------------------------------------------------------------------------- router
st.sidebar.title("Customer Churn DAV")
page = st.sidebar.radio("Navigate", PAGES, key="page")
st.sidebar.caption("Intelligent Customer Churn Prediction using Data Analytics and Hybrid Machine Learning")
{"1. Overview": page_overview, "2. EDA": page_eda, "3. Model Comparison": page_comparison,
 "4. Model Evaluation": page_evaluation, "5. Churn Prediction": page_prediction,
 "6. Explainability": page_explainability}[page]()

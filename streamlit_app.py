"""
Customer Churn Prediction - Streamlit web app.

Online dashboard + predictor for the Telco churn project. Reuses the trained
models and preprocessing from the repo (Flask app + src/ pipeline) and adds:

    * Model performance dashboard (KPIs, confusion matrices, bar charts)
    * Single-customer churn predictor with interpretation
    * Batch CSV upload with downloadable results
    * About / methodology

Deploy target: Streamlit Community Cloud (share.streamlit.io), which serves
this file directly from the GitHub repository.
"""
import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page config (top of script, avoids visual blink)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon=":material/analytics:",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"

MODEL_NAMES = ["linear_regression", "random_forest", "xgboost"]
MODEL_DISPLAY = {
    "linear_regression": "Linear Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}


# ---------------------------------------------------------------------------
# Cached artifact loading
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading churn models...")
def load_models():
    """Load the three saved pipelines once per session."""
    return {name: joblib.load(MODEL_DIR / f"{name}.pkl") for name in MODEL_NAMES}


@st.cache_data(show_spinner=False)
def load_artifacts():
    """Load metrics / thresholds / importances JSON files."""
    def _read(name):
        return json.loads((MODEL_DIR / name).read_text(encoding="utf-8"))

    return {
        "metrics": _read("model_metrics.json"),
        "thresholds": _read("thresholds.json"),
        "importance": _read("feature_importance.json"),
    }


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------
def predict_customer(model_name: str, pipeline, df: pd.DataFrame) -> tuple:
    """Return (probability_array, hard_prediction_array, threshold)."""
    model = pipeline.named_steps["model"]
    if hasattr(model, "predict_proba"):
        proba = pipeline.predict_proba(df)[:, 1]
    else:  # Linear regression -> clip the raw score into [0,1]
        proba = np.clip(pipeline.predict(df), 0, 1)
    threshold = float(load_artifacts()["thresholds"].get(model_name, {}).get("threshold", 0.5))
    return proba, (proba >= threshold).astype(int), threshold


def linear_local_drivers(pipeline, df: pd.DataFrame, top: int = 6) -> list:
    """Coefficient-weighted feature contributions (interpretation for LR)."""
    prep = pipeline.named_steps["prep"]
    cleaned = pipeline.named_steps["clean"].transform(df)
    Xt = prep.transform(cleaned)
    coefs = np.ravel(pipeline.named_steps["model"].coef_)

    num_names = prep.named_transformers_["num"].get_feature_names_out().tolist()
    bin_names = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    cat_names = prep.named_transformers_["cat"].get_feature_names_out().tolist()
    names = num_names + bin_names + cat_names

    contributions = [
        {"feature": str(n), "value": round(float(c), 4)}
        for n, c in zip(names, coefs * Xt[0])
        if abs(c) > 1e-4
    ]
    contributions.sort(key=lambda x: x["value"], reverse=True)
    return contributions[:top]


def build_row(form: dict) -> pd.DataFrame:
    """Assemble a raw one-row dataframe from the form widgets."""
    raw = {
        "gender": form["gender"],
        "SeniorCitizen": int(form["SeniorCitizen"]),
        "Partner": form["Partner"],
        "Dependents": form["Dependents"],
        "tenure": int(form["tenure"]),
        "PhoneService": form["PhoneService"],
        "MultipleLines": form["MultipleLines"],
        "InternetService": form["InternetService"],
        "OnlineSecurity": form["OnlineSecurity"],
        "OnlineBackup": form["OnlineBackup"],
        "DeviceProtection": form["DeviceProtection"],
        "TechSupport": form["TechSupport"],
        "StreamingTV": form["StreamingTV"],
        "StreamingMovies": form["StreamingMovies"],
        "Contract": form["Contract"],
        "PaperlessBilling": form["PaperlessBilling"],
        "PaymentMethod": form["PaymentMethod"],
        "MonthlyCharges": float(form["MonthlyCharges"]),
        "TotalCharges": float(form["TotalCharges"]),
    }
    return pd.DataFrame([raw])


def risk_label(proba: float) -> str:
    if proba >= 0.7:
        return "High"
    if proba >= 0.45:
        return "Medium"
    return "Low"


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_dashboard():
    models = load_models()
    art = load_artifacts()
    metrics = art["metrics"]
    importance = art["importance"]

    st.markdown(
        "### :material/query_stats: Model performance dashboard"
    )
    st.caption(
        "Hold-out test set: 1,407 customers (26.6% churn rate) "
        "· models trained on 5,625 customers"
    )

    # KPI row
    best = max(metrics, key=lambda m: metrics[m]["roc_auc"])
    with st.container(horizontal=True):
        st.metric(
            ":material/database: Dataset size",
            "7,032",
            "records after cleaning",
            border=True,
        )
        st.metric(
            ":material/pie_chart: Churn rate", "26.6%", "imbalanced classes", border=True,
        )
        st.metric(
            ":material/emoji_events: Best ROC-AUC",
            f"{metrics[best]['roc_auc']:.3f}",
            MODEL_DISPLAY[best],
            border=True,
        )
        st.metric(
            ":material/network_node: Models deployed", "3", "LR · RF · XGBoost", border=True,
        )

    # Comparison table + accuracy bars
    col1, col2 = st.columns([3, 2])
    with col1:
        with st.container(border=True):
            st.markdown("**Model comparison**")
            comp = [
                {
                    "Model": MODEL_DISPLAY[m],
                    "Accuracy": f"{metrics[m]['accuracy']*100:.1f}%",
                    "Precision": f"{metrics[m]['precision']*100:.1f}%",
                    "Recall (churn)": f"{metrics[m]['recall']*100:.1f}%",
                    "F1": f"{metrics[m]['f1_score']:.3f}",
                    "ROC-AUC": f"{metrics[m]['roc_auc']:.3f}",
                }
                for m in MODEL_NAMES
            ]
            st.dataframe(pd.DataFrame(comp), hide_index=True, width="stretch")

            st.markdown("**Accuracy comparison**", )
            st.bar_chart(
                pd.DataFrame(
                    {"Model": [MODEL_DISPLAY[m] for m in MODEL_NAMES],
                     "Accuracy (%)": [metrics[m]["accuracy"] * 100 for m in MODEL_NAMES]}
                ).set_index("Model")
            )

    with col2:
        with st.container(border=True):
            st.markdown("**Confusion matrices**")
            tabs = st.tabs([MODEL_DISPLAY[m] for m in MODEL_NAMES])
            for tab, m in zip(tabs, MODEL_NAMES):
                with tab:
                    st.dataframe(
                        pd.DataFrame(
                            metrics[m]["confusion_matrix"],
                            index=["Actual No", "Actual Yes"],
                            columns=["Pred No", "Pred Yes"],
                        ),
                        width="stretch",
                    )
            st.caption("Row = actual, column = predicted. Higher diagonal = better.")

    # Feature importances
    st.markdown("### :material/account_tree: Top churn drivers")
    imp_cols = st.columns(3)
    for col, m in zip(imp_cols, MODEL_NAMES):
        with col:
            with st.container(border=True):
                st.markdown(f"**{MODEL_DISPLAY[m]}**")
                top = importance["per_model"][m]["top_features"][:6]
                st.dataframe(
                    pd.DataFrame(top),
                    hide_index=True,
                    width="stretch",
                )

    st.markdown("""
    :gray-background[**Note on the 85–92% target:** on this public Telco dataset 75–80% accuracy is
    the realistic state of the art — the trivial "predict everyone stays" baseline scores 73.4%.
    All three models beat the baseline clearly (AUC 0.82–0.84). **Random Forest** is the best choice
    for a retention program: it finds 76% of future churners before they leave.]
    """)


def page_predict():
    models = load_models()
    st.markdown("### :material/person_search: Single-customer prediction")
    st.caption(
        "Fill the customer profile. All three models score the record and explain their reasoning."
    )

    with st.form("customer_form", border=True):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**:material/badge: Customer profile**")
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior citizen", [0, 1], format_func=lambda v: "No (0)" if v == 0 else "Yes (1)")
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            tenure = st.number_input("Tenure (months)", 0, 72, 12)

        with c2:
            st.markdown("**:material/router: Services**")
            phone = st.selectbox("Phone service", ["Yes", "No"])
            multiple = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
            internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
            security = st.selectbox("Online security", ["No", "Yes", "No internet service"])
            backup = st.selectbox("Online backup", ["No", "Yes", "No internet service"])
            device = st.selectbox("Device protection", ["No", "Yes", "No internet service"])

        with c3:
            st.markdown("**:material/description: Contract & services (cont.)**")
            support = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
            tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            movies = st.selectbox("Streaming movies", ["No", "Yes", "No internet service"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless billing", ["Yes", "No"])
            payment = st.selectbox(
                "Payment method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            )

        c4, c5, _ = st.columns(3)
        with c4:
            monthly = st.number_input("Monthly charges", 0.0, 300.0, 65.0, 0.01)
        with c5:
            total = st.number_input("Total charges", 0.0, 15000.0, 800.0, 0.01)

        submitted = st.form_submit_button(
            ":material/insights: Predict churn risk", type="primary", width="stretch"
        )

    if not submitted:
        st.info(
            "Submit the form to score this customer with all three models.",
            icon=":material/info:",
        )
        return

    form = {
        "gender": gender, "SeniorCitizen": senior, "Partner": partner,
        "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
        "MultipleLines": multiple, "InternetService": internet,
        "OnlineSecurity": security, "OnlineBackup": backup,
        "DeviceProtection": device, "TechSupport": support,
        "StreamingTV": tv, "StreamingMovies": movies, "Contract": contract,
        "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": monthly, "TotalCharges": total,
    }
    row = build_row(form)

    st.space("small")
    st.markdown("#### Verdicts")
    results = []
    for name in MODEL_NAMES:
        proba, hard, threshold = predict_customer(name, models[name], row)
        p0 = float(proba[0])
        results.append(
            {"model": name, "prob": p0, "hard": int(hard[0]), "threshold": threshold,
             "risk": risk_label(p0)}
        )

    cols = st.columns(len(results), border=True)
    for col, r in zip(cols, results):
        with col:
            if r["hard"] == 1:
                st.badge("CHURN", icon=":material/warning:", color="red")
            else:
                st.badge("NO CHURN", icon=":material/check:", color="green")
            st.markdown(f"**{MODEL_DISPLAY[r['model']]}**")
            st.metric("Churn probability", f"{r['prob']*100:.1f}%", border=True)
            st.progress(int(r["prob"] * 100))
            st.caption(f"Risk: **{r['risk']}** · decision threshold {r['threshold']:.2f}")
            st.caption(f"Acc {load_artifacts()['metrics'][r['model']]['accuracy']*100:.0f}% · "
                       f"AUC {load_artifacts()['metrics'][r['model']]['roc_auc']:.3f}")

    # Consensus callout
    churn_votes = sum(1 for r in results if r["hard"] == 1)
    if churn_votes >= 2:
        st.warning(
            f"**{churn_votes} of 3 models flag churn risk.** This customer is highly likely to churn — "
            "act now: retention offer, plan change, support outreach.",
            icon=":material/priority_high:",
        )
    elif churn_votes == 1:
        st.info(
            "**1 of 3 models flags churn risk.** Monitor this customer closely — review billing, "
            "contract and service satisfaction.",
            icon=":material/manage_search:",
        )
    else:
        st.success(
            "**No model flags churn risk.** Strong retention signals — reinforce with loyalty perks.",
            icon=":material/thumb_up:",
        )

    # Interpretation
    st.markdown("#### :material/psychology: Why — interpretation")
    art = load_artifacts()
    lr_drivers = linear_local_drivers(models["linear_regression"], row)
    e1, e2 = st.columns(2)
    with e1:
        with st.container(border=True):
            st.markdown("**Linear regression — local drivers**")
            st.caption("Per-feature contribution to this customer's churn score.")
            if lr_drivers:
                for d in lr_drivers:
                    color = "red" if d["value"] > 0 else ("green" if d["value"] < 0 else "gray")
                    arrow = "▲" if d["value"] > 0 else "▼" if d["value"] < 0 else "•"
                    st.markdown(
                        f":{color}[{arrow} {d['feature']}]  `{d['value']:+.4f}`"
                    )
                st.caption("▲ pushes toward churn · ▼ pushes toward retention")
            else:
                st.caption("No significant contributors for this record.")
    with e2:
        with st.container(border=True):
            st.markdown("**Tree models — top global drivers**")
            st.caption("Most important features the models rely on across all customers.")
            t1, t2 = st.columns(2)
            for tcol, m in zip([t1, t2], ["random_forest", "xgboost"]):
                with tcol:
                    st.markdown(f"**{MODEL_DISPLAY[m]}**")
                    for f in art["importance"]["per_model"][m]["top_features"][:6]:
                        st.markdown(f"- {f['feature']}")


def page_batch():
    models = load_models()
    st.markdown("### :material/upload_file: Batch CSV prediction")
    st.caption(
        "Upload a CSV with the same columns as the training data (customerID optional, "
        "a Churn column is ignored). Each row is scored by all three models plus an ensemble verdict."
    )

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV to get started.", icon=":material/upload:")
        return

    df = pd.read_csv(io.BytesIO(uploaded.getvalue()))
    st.success(f"Loaded {len(df)} rows × {len(df.columns)} columns.", icon=":material/check_circle:")

    with st.spinner("Scoring all customers with 3 models..."):
        out = df.copy()
        for name in MODEL_NAMES:
            proba, hard, _ = predict_customer(name, models[name], df)
            out[f"{name}_churn_probability"] = np.round(proba, 4)
            out[f"{name}_prediction"] = hard

        votes = out[[f"{name}_prediction" for name in MODEL_NAMES]].sum(axis=1)
        out["ensemble_agreement"] = votes
        out["ensemble_verdict"] = np.where(votes >= 2, "Churn", "No Churn")

    churn_count = int((out["ensemble_verdict"] == "Churn").sum())
    with st.container(horizontal=True):
        st.metric("Customers scored", len(out), border=True)
        st.metric("Predicted churn", churn_count, border=True)
        st.metric("Churn rate (predicted)", f"{churn_count/len(out)*100:.1f}%", border=True)

    pred_cols = [c for c in out.columns if "prob" in c or "pred" in c or "verdict" in c or "agreement" in c]
    st.markdown("**Preview**")
    st.dataframe(out[[c for c in out.columns if c in [*pred_cols, "tenure", "Contract", "MonthlyCharges"]]],
                 hide_index=True, width="stretch")

    csv_bytes = out.to_csv(index=False).encode("utf-8")
    st.download_button(
        ":material/download: Download full results",
        csv_bytes,
        file_name="churn_predictions.csv",
        mime="text/csv",
        type="primary",
    )


def page_about():
    st.markdown("### :material/school: About this project")
    st.markdown(
        """
**Problem.** Customer attrition costs subscription businesses (telecom, banking, OTT — Jio, Airtel,
HDFC, ICICI, Netflix, Hotstar) over ₹10,000 crore a year. Rule-based retention systems miss complex
behavioural patterns, and generic models ignore localised drivers. This project delivers
production-ready, interpretable churn prediction.

**Dataset.** IBM Telco Customer Churn — 7,032 usable records, 20 features, 26.6% churn.
Features cover demographics, services, contract type and billing behaviour (a proxy for payment
irregularity and usage trends).

**Models.** Linear Regression (baseline), Random Forest, XGBoost — each saved as a self-contained
preprocessing + model pipeline with engineered features (`avg_charge_per_month`, `tenure_log`) and
accuracy-optimised decision thresholds.

**How to interpret results.** The dashboard compares accuracy/precision/recall/F1/AUC. For a single
customer, the Linear Regression model lists *local* drivers (its coefficients × this customer's
values); tree models list their top *global* importance features.

**Honesty about accuracy.** That 85–92% headline is not honestly reachable on this public dataset —
the trivial baseline is 73.4% accuracy and the models reach 75–80% while beating the baseline on
ROC-AUC (0.82–0.84). Random Forest is recommended for retention teams (finds 76% of true churners).

**Links**
- GitHub: [prabuMessi10/customer-churn-prediction](https://github.com/prabuMessi10/customer-churn-prediction)
- Stack: Streamlit · scikit-learn · XGBoost · Flask (companion API) · pandas
        """
    )

    with st.container(border=True):
        st.markdown("**Model performance (hold-out test set)**")
        art = load_artifacts()
        rows = []
        for m in MODEL_NAMES:
            met = art["metrics"][m]
            rows.append(
                {
                    "Model": MODEL_DISPLAY[m],
                    "Accuracy": f"{met['accuracy']*100:.1f}%",
                    "Recall (churn)": f"{met['recall']*100:.1f}%",
                    "F1": f"{met['f1_score']:.3f}",
                    "ROC-AUC": f"{met['roc_auc']:.3f}",
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# Navigation shell
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### :material/analytics: Churn Predictor")
    st.caption("Telecom · Banking · OTT · Subscription services")
    page = st.radio(
        "Navigate",
        ["Model dashboard", "Predict a customer", "Batch upload", "About"],
        label_visibility="collapsed",
    )
    st.space("medium")
    st.caption("Source: [GitHub](https://github.com/prabuMessi10/customer-churn-prediction)")

if page == "Model dashboard":
    page_dashboard()
elif page == "Predict a customer":
    page_predict()
elif page == "Batch upload":
    page_batch()
else:
    page_about()
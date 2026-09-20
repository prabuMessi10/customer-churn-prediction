# Customer Churn Prediction — Machine Learning Project

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://customer-churn-prediction-on.streamlit.app)

**🔗 Live demo: https://customer-churn-prediction-on.streamlit.app** (hosted on Streamlit Community Cloud)

Predict customer churn for subscription businesses (telecom / banking / OTT) using
**Linear Regression**, **Random Forest** and **XGBoost**, served through a deployed
Streamlit dashboard.

## Problem addressed

- Severe financial losses (₹10,000 cr+) from churn in telecom, banking and OTT businesses.
- Rule-based retention systems miss complex behavioural patterns.
- Generic models ignore localised behavioural drivers.
- Need for production-ready, interpretable, high-accuracy models.

## Project structure

This repository is kept minimal on purpose — it contains **only what the hosted
Streamlit app needs to run**:

```
customer-churn-prediction/
├── .streamlit/
│   └── config.toml              # dark indigo theme
├── model/                       # persisted trained artifacts
│   ├── linear_regression.pkl    # full pipeline (cleaning + scaling + model)
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── preprocessor.pkl         # fitted ColumnTransformer
│   ├── model_metrics.json       # test-set evaluation for all models
│   ├── feature_importance.json  # top churn drivers per model
│   ├── thresholds.json          # accuracy-optimised decision thresholds
│   └── feature_columns.json     # schema for the prediction form
├── .gitignore
├── requirements.txt
├── streamlit_app.py             # Streamlit entry point (4 pages)
└── README.md
```

## The Streamlit app

Four pages, all sharing the same trained models and preprocessing:

1. **Model dashboard** – KPIs, side-by-side accuracy / precision / recall / F1 / AUC
   and confusion matrices for all three models.
2. **Predict a customer** – fill the form, get a verdict from each model, churn
   probability, risk level, consensus note and an interpretation of the drivers.
3. **Batch upload** – submit a CSV; download results with probabilities, per-model
   verdicts and an ensemble (majority-vote) prediction.
4. **About** – methodology and an honest note on achievable accuracy.

## Setup & run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
# open http://localhost:8501
```

## Deploy online with Streamlit Community Cloud (free)

The app is already deployed: **https://customer-churn-prediction-on.streamlit.app**

To redeploy or deploy a fork:

1. Push this repo to GitHub (currently at `prabuMessi10/customer-churn-prediction`).
2. Go to https://share.streamlit.io and **sign in with GitHub** (the account that owns the repo).
3. Click **Create app** → choose the repo, branch `main`, main file `streamlit_app.py`.
4. Click **Deploy** — Streamlit installs `requirements.txt`, clones the models and serves
   the app at a public `*.streamlit.app` URL. Every push to `main` auto-redeploys.

## Model performance (hold-out test set)

| Model             | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------------------|----------|-----------|--------|-----|---------|
| Linear Regression | 79.8%    | 66.1%     | 49.5%  | 0.57| 0.837   |
| Random Forest     | 75.8%    | 53.2%     | 76.2%  | 0.63| 0.833   |
| XGBoost           | 77.2%    | 56.9%     | 58.6%  | 0.58| 0.817   |
| Majority baseline | 73.4%    | —         | 0%     | —   | 0.5     |

**Note on the 85–92% accuracy target:** on this publicly available Telco dataset 75–80% accuracy
is the realistic state of the art; the trivial "predict everyone stays" baseline is 73.4%.
All three models clearly beat the baseline (AUC 0.82–0.84). Random Forest offers the best
balance for a retention program because it finds 76% of future churners before they leave.

## Feature engineering

Derived features (created inside the saved pipeline, so the web app needs no extra code):

- `avg_charge_per_month` = TotalCharges / tenure
- `tenure_log` = log1p(tenure)
- Binary columns (gender, Partner, Dependents, PhoneService, PaperlessBilling) mapped to 0/1
  and passed through; numeric columns standardised; categorical columns one-hot encoded.

## Contact

Built by **Subash Acharya D (24CSR306)**, **Thannasi Prabu R (24CSR321)** and **Vikram S (24CSR345)**.
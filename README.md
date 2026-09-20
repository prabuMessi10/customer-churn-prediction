# Customer Churn Prediction - Full-Stack ML Project

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://customer-churn-prediction-on.streamlit.app)

**🔗 Live demo: https://customer-churn-prediction-on.streamlit.app** (Streamlit · hosted on Streamlit Community Cloud)

Predict customer churn for subscription businesses (telecom / banking / OTT) using
**Linear Regression**, **Random Forest** and **XGBoost**, served through a Flask web app
and a deployed Streamlit dashboard.

## Problem addressed

- Severe financial losses (₹10,000 cr+) from churn in Jio/Airtel, HDFC/ICICI, Netflix/Hotstar.
- Rule-based retention systems miss complex behavioural patterns.
- Generic models ignore localised behavioural drivers.
- Need for production-ready, interpretable, high-accuracy models.

## Project structure

```
final project/
├── dataset/
│   └── Telco_Customer_Churn.csv      # raw dataset (7,043 rows, 21 cols)
├── model/                            # persisted trained artifacts
│   ├── linear_regression.pkl         # full pipeline (cleaning+scaling+model)
│   ├── random_forest.pkl             #            (accuracy 76%, recall 76%)
│   ├── xgboost.pkl                   #            (accuracy 77%, recall 59%)
│   ├── preprocessor.pkl              # fitted ColumnTransformer
│   ├── model_metrics.json            # test-set evaluation for all models
│   ├── feature_importance.json       # top churn drivers per model
│   ├── thresholds.json               # accuracy-optimised decision thresholds
│   └── feature_columns.json          # schema for the web form
├── src/
│   ├── preprocess.py                 # cleaning + feature engineering + transformer
│   ├── train_models.py               # trains & exports the three models
│   ├── tune_models.py                # hyper-parameter sweep (exploration)
│   └── optimise_thresholds.py        # finds best decision thresholds
├── templates/
│   ├── index.html                    # dashboard + prediction form
│   ├── results.html                  # model verdicts + interpretation
│   └── batch.html                    # CSV batch upload
├── static/
│   └── style.css
├── app.py                            # Flask application
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Retrain the models (optional)

```bash
python -m src.train_models            # trains all 3 models, saves to model/
python -m src.optimise_thresholds     # recompute accuracy-optimised thresholds
```

## Run the web app (Flask)

```bash
python app.py
# open http://127.0.0.1:5000
```

The app provides:

1. **Model dashboard** – side-by-side accuracy / precision / recall / F1 / AUC and confusion matrices.
2. **Single prediction** – fill the form, get a verdict from each model, churn probability,
   risk level, consensus note and an interpretation of the drivers.
3. **Batch upload** – submit a CSV; download results with probabilities, per-model verdicts and
   an ensemble (majority-vote) prediction.
4. **JSON API** – `POST /api/predict` with a JSON customer record.

## Run the web app (Streamlit)

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
# open http://localhost:8501
```

### Deploy online with Streamlit Community Cloud (free)

The app is already deployed: **https://customer-churn-prediction-on.streamlit.app**

To redeploy or deploy a fork:

1. Push this repo to GitHub (already at `prabuMessi10/customer-churn-prediction`).
2. Go to https://share.streamlit.io and **sign in with GitHub** (the account that owns the repo).
3. Click **Create app** → choose the repo, branch `main`, main file `streamlit_app.py`.
4. Click **Deploy** — Streamlit installs `requirements.txt`, clones the models and serves the app
   at a public `*.streamlit.app` URL.

The Streamlit app includes the same four capabilities (dashboard, single prediction, batch upload,
about) plus live model interpretation.

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
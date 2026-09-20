"""
Customer Churn Prediction - Flask web application.

Runs three trained models (Linear Regression, Random Forest, XGBoost) on
customer records entered through the browser form, shows the churn risk from
every model, and explains the drivers behind the prediction.
"""
import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load artifacts once at startup
# ---------------------------------------------------------------------------
MODELS = {
    "linear_regression": joblib.load(MODEL_DIR / "linear_regression.pkl"),
    "random_forest": joblib.load(MODEL_DIR / "random_forest.pkl"),
    "xgboost": joblib.load(MODEL_DIR / "xgboost.pkl"),
}
PREPROCESSOR = joblib.load(MODEL_DIR / "preprocessor.pkl")
METRICS = json.loads((MODEL_DIR / "model_metrics.json").read_text(encoding="utf-8"))
IMPORTANCE = json.loads((MODEL_DIR / "feature_importance.json").read_text(encoding="utf-8"))
THRESHOLDS = json.loads((MODEL_DIR / "thresholds.json").read_text(encoding="utf-8"))
FEATURES = json.loads((MODEL_DIR / "feature_columns.json").read_text(encoding="utf-8"))

MODEL_DISPLAY = {
    "linear_regression": "Linear Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}

# Coefficient column order for the linear-model local explanation
LR_COEF_NAMES = (
    PREPROCESSOR.named_transformers_["num"].get_feature_names_out()
    if hasattr(PREPROCESSOR.named_transformers_["num"], "get_feature_names_out")
    else None
)

MODEL_NAMES = list(MODELS.keys())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def probability_for(model_name: str, pipeline, df: pd.DataFrame) -> tuple:
    """Return (risk_probability_array, hard_prediction_array) for a dataframe."""
    model = pipeline.named_steps["model"]
    if hasattr(model, "predict_proba"):
        proba = pipeline.predict_proba(df)[:, 1]
    else:  # LinearRegression -> clip the raw score
        proba = np.clip(pipeline.predict(df), 0, 1)
    threshold = float(THRESHOLDS.get(model_name, {}).get("threshold", 0.5))
    return proba, (proba >= threshold).astype(int), threshold


def linear_explanation(pipeline, row: pd.DataFrame) -> list:
    """Local feature contributions for the linear regression model."""
    cleaned = pipeline.named_steps["clean"].transform(row)
    Xt = pipeline.named_steps["prep"].transform(cleaned)
    coefs = np.ravel(pipeline.named_steps["model"].coef_)

    # Reconstruct feature names: numeric + binary + one-hot categories
    prep = pipeline.named_steps["prep"]
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
    return contributions


def top_drivers(model_name: str):
    """Global top-5 features for tree models (interpretability fallback)."""
    imp = IMPORTANCE["per_model"].get(model_name, {}).get("top_features", [])
    return imp[:5]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template(
        "index.html",
        metrics=METRICS,
        display=MODEL_DISPLAY,
        feature_columns=FEATURES,
        threshold_defaults={k: v.get("threshold", 0.5) for k, v in THRESHOLDS.items()},
        drivers={k: top_drivers(k) for k in MODEL_NAMES},
        lift=IMPORTANCE.get("lift_over_baseline", {}),
    )


@app.route("/predict", methods=["POST"])
def predict():
    """Single-customer prediction from the browser form."""
    form = request.form.to_dict()

    # Build one-row DataFrame matching the raw dataset schema
    raw_row = {
        "gender": form.get("gender", "Female"),
        "SeniorCitizen": int(form.get("SeniorCitizen", 0)),
        "Partner": form.get("Partner", "No"),
        "Dependents": form.get("Dependents", "No"),
        "tenure": int(form.get("tenure", 0)),
        "PhoneService": form.get("PhoneService", "Yes"),
        "MultipleLines": form.get("MultipleLines", "No"),
        "InternetService": form.get("InternetService", "No"),
        "OnlineSecurity": form.get("OnlineSecurity", "No"),
        "OnlineBackup": form.get("OnlineBackup", "No"),
        "DeviceProtection": form.get("DeviceProtection", "No"),
        "TechSupport": form.get("TechSupport", "No"),
        "StreamingTV": form.get("StreamingTV", "No"),
        "StreamingMovies": form.get("StreamingMovies", "No"),
        "Contract": form.get("Contract", "Month-to-month"),
        "PaperlessBilling": form.get("PaperlessBilling", "Yes"),
        "PaymentMethod": form.get("PaymentMethod", "Electronic check"),
        "MonthlyCharges": float(form.get("MonthlyCharges", 0) or 0),
        "TotalCharges": float(form.get("TotalCharges", 0) or 0),
    }
    row = pd.DataFrame([raw_row])

    results = []
    for name in MODEL_NAMES:
        proba, hard, th = probability_for(name, MODELS[name], row)
        p0, h0 = float(proba[0]), int(hard[0])
        results.append(
            {
                "model": name,
                "display": MODEL_DISPLAY[name],
                "probability": round(p0, 4),
                "churn_percent": round(p0 * 100, 1),
                "prediction": "Churn" if h0 else "No Churn",
                "threshold": th,
                "risk": (
                    "High"
                    if p0 >= 0.7
                    else "Medium"
                    if p0 >= 0.45
                    else "Low"
                ),
                "metrics": METRICS.get(name, {}),
            }
        )

    # Best model = highest probability of capturing true churn (by ROC-AUC)
    best = max(results, key=lambda r: r["metrics"].get("roc_auc", 0))

    # Local explanation for the linear model
    explanation = {
        "linear_regression": linear_explanation(MODELS["linear_regression"], row)[:6],
        "random_forest": top_drivers("random_forest"),
        "xgboost": top_drivers("xgboost"),
    }

    return render_template(
        "results.html",
        input=raw_row,
        results=results,
        best=best,
        explanation=explanation,
        display=MODEL_DISPLAY,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API: predict for a single customer record."""
    data = request.get_json(force=True)
    try:
        row = pd.DataFrame([data])
        out = {}
        for name in MODEL_NAMES:
            proba, hard, th = probability_for(name, MODELS[name], row)
            out[name] = {"churn_probability": float(proba[0]), "prediction": int(hard[0]), "threshold": th}
        return jsonify({"success": True, "predictions": out})
    except Exception as exc:  # noqa: BLE001 - surface errors to API callers
        return jsonify({"success": False, "error": str(exc)}), 400


@app.route("/batch", methods=["GET", "POST"])
def batch():
    """Batch prediction: upload a CSV with the same schema, download results."""
    if request.method == "POST":
        file = request.files.get("file")
        if file is None or not file.filename:
            return render_template("batch.html", error="Please choose a CSV file.")

        df = pd.read_csv(io.BytesIO(file.read()))
        results_df = df.copy()
        for name in MODEL_NAMES:
            proba, hard, th = probability_for(name, MODELS[name], df)
            results_df[f"{name}_churn_probability"] = np.round(proba, 4)
            results_df[f"{name}_prediction"] = hard

        # Ensemble agreement (majority vote among the three models)
        votes = results_df[[f"{name}_prediction" for name in MODEL_NAMES]].sum(axis=1)
        results_df["ensemble_agreement"] = votes
        results_df["ensemble_verdict"] = np.where(
            votes >= 2, "Churn", "No Churn"
        )

        summary = {
            "rows": int(len(results_df)),
            "predicted_churn": int((results_df["ensemble_verdict"] == "Churn").sum()),
            "predicted_churn_pct": round(
                float((results_df["ensemble_verdict"] == "Churn").mean() * 100), 1
            ),
        }

        output = io.BytesIO()
        results_df.to_csv(output, index=False)
        output.seek(0)
        return send_file(
            output,
            as_attachment=True,
            download_name="churn_predictions.csv",
            mimetype="text/csv",
        )

    return render_template("batch.html")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
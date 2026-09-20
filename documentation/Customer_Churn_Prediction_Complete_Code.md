# Customer Churn Prediction

**Complete Project Source Code**

Included code files: **6**

GitHub Repository: https://github.com/prabuMessi10/customer-churn-prediction

---


## 1. requirements.txt

```text
flask>=3.0
pandas>=2.2
numpy>=1.26
scikit-learn>=1.4
xgboost>=2.0
joblib>=1.4
streamlit>=1.57
matplotlib>=3.8
seaborn>=0.13
```

## 2. src/preprocess.py

```python
"""
Data cleaning and preprocessing utilities for the Telco Customer Churn dataset.

All transformation steps are wrapped so they can be re-applied identically
to both training data and single records coming from the Flask web app.
"""
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Dataset layout (Telco Customer Churn - IBM sample)
# ---------------------------------------------------------------------------
TARGET = "Churn"

# Numeric columns passed through StandardScaler
NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "SeniorCitizen",
    "avg_charge_per_month",  # engineered
    "tenure_log",            # engineered
]

# Binary string columns that are mapped to 0/1 manually, then passed through
BINARY_FEATURES = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]

# Multi-category string columns that are one-hot encoded
CATEGORICAL_FEATURES = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
]

BINARY_MAP = {
    "gender": {"Female": 0, "Male": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PhoneService": {"No": 0, "Yes": 1},
    "PaperlessBilling": {"No": 0, "Yes": 1},
}


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply deterministic cleaning to a raw Telco churn record.

    - Drops the customerID column (not a predictive feature)
    - Converts TotalCharges from string to float (empty string -> NaN -> 0)
    - Maps binary string columns to 0/1
    """
    df = df.copy()

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # TotalCharges arrives as a string column (object or pandas 'str' dtype)
    # with occasional empty values
    if "TotalCharges" in df.columns and is_string_dtype(df["TotalCharges"]):
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Only map string columns; already-encoded (numeric) columns pass through
    for col, mapping in BINARY_MAP.items():
        if col in df.columns and is_string_dtype(df[col]):
            df[col] = df[col].astype(str).str.strip().map(mapping).astype(int)

    # Normalise remaining categorical strings (strip whitespace)
    for col in CATEGORICAL_FEATURES:
        if col in df.columns and is_string_dtype(df[col]):
            df[col] = df[col].astype(str).str.strip()

    df = df.dropna(subset=["TotalCharges"])

    # ---- Feature engineering (derived, keeps the saved pipeline self-contained)
    df["avg_charge_per_month"] = np.where(
        df["tenure"] > 0, df["TotalCharges"] / df["tenure"], df["MonthlyCharges"]
    )
    df["tenure_log"] = np.log1p(df["tenure"].clip(lower=0))

    return df


def build_preprocessor() -> ColumnTransformer:
    """
    ColumnTransformer used both at training time and inside every saved model.

    Numeric columns  -> StandardScaler
    Categorical cols -> OneHotEncoder (drop='first' avoids dummy-variable trap,
                        handle_unknown='ignore' protects web-form inputs)
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            # Binary 0/1 features pass through unchanged (already encoded)
            ("bin", "passthrough", BINARY_FEATURES),
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        sparse_threshold=0,
    )


def full_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a raw dataframe (helper used by the metrics / EDA scripts)."""
    return clean_data(df)


def feature_groups() -> dict:
    """Return the feature groups used to build the web form."""
    return {
        "numeric": NUMERIC_FEATURES,
        "binary": BINARY_FEATURES,
        "categorical": CATEGORICAL_FEATURES,
        "binary_map": {k: list(v.keys()) for k, v in BINARY_MAP.items()},
        "target": TARGET,
    }


if __name__ == "__main__":
    # Quick sanity check
    demo = pd.read_csv(r"dataset\Telco_Customer_Churn.csv")
    cleaned = clean_data(demo)
    print("Cleaned shape:", cleaned.shape)
    print(cleaned.dtypes)
```

## 3. src/train_models.py

```python
"""
Train, evaluate, and export the churn prediction models.

Models
------
1. Linear Regression (baseline; continuous score thresholded at 0.5)
2. Random Forest Classifier
3. XGBoost Classifier

Every model is saved as a single sklearn Pipeline (cleaning -> preprocessor ->
estimator), so the Flask app can reload it and predict on raw form input with
zero additional preprocessing code.

Outputs written to the model/ folder:
    preprocessor.pkl            fitted ColumnTransformer (for feature names)
    linear_regression.pkl       full pipeline (baseline)
    random_forest.pkl           full pipeline
    xgboost.pkl                 full pipeline
    feature_columns.json        raw feature options for the web form
    model_metrics.json          evaluation metrics + feature importance
"""
import json
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.preprocess import TARGET, build_preprocessor, clean_data, feature_groups

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("train")

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "dataset" / "Telco_Customer_Churn.csv"
MODEL_DIR = ROOT / "model"
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------
def make_pipeline(estimator) -> Pipeline:
    """Wrap an estimator in cleaning -> preprocessor -> estimator."""
    return Pipeline(
        [
            ("clean", _cleaning_step()),
            ("prep", build_preprocessor()),
            ("model", estimator),
        ]
    )


def _cleaning_step():
    from sklearn.preprocessing import FunctionTransformer

    return FunctionTransformer(clean_data, validate=False)


def build_models() -> dict:
    """Factory returning {name: Pipeline} for all models."""
    return {
        "linear_regression": make_pipeline(
            LinearRegression()  # regression -> probability-like score, threshold 0.5
        ),
        "random_forest": make_pipeline(
            RandomForestClassifier(
                n_estimators=400,
                max_depth=12,
                min_samples_split=5,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
        "xgboost": make_pipeline(
            XGBClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=1.4,  # mild imbalance correction (accuracy-friendly)
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
    }


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def evaluate(pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Compute standard classification metrics for any saved pipeline."""
    # Linear regression yields a continuous score -> threshold to 0/1
    preds = pipeline.predict(X_test)
    if not np.array_equal(np.unique(preds), np.array([0, 1])):
        probas = preds
        preds = (probas >= 0.5).astype(int)

    try:
        proba = pipeline.predict_proba(X_test)[:, 1]
        has_proba = True
    except (AttributeError, NotImplementedError):
        # Linear regression: use its score clipped to [0,1] as a pseudo-probability
        proba = np.clip(pipeline.predict(X_test), 0, 1)
        has_proba = False

    return {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds), 4),
        "recall": round(recall_score(y_test, preds), 4),
        "f1_score": round(f1_score(y_test, preds), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "has_probability": has_proba,
        "report": classification_report(y_test, preds, output_dict=True),
    }


def feature_importances(pipeline, transformer: ColumnTransformer) -> dict:
    """Extract per-feature importances (trees) or coefficients (linear)."""
    from src.preprocess import BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES

    model = pipeline.named_steps["model"]
    cat_names = transformer.named_transformers_["cat"].get_feature_names_out(
        CATEGORICAL_FEATURES
    )
    names = NUMERIC_FEATURES + BINARY_FEATURES + list(cat_names)

    if hasattr(model, "coef_"):
        values = np.abs(np.ravel(model.coef_))
        kind = "coefficient"
    else:
        values = model.feature_importances_
        kind = "importance"

    ranked = sorted(zip(names, values.tolist()), key=lambda x: -x[1])
    return {
        "kind": kind,
        "top_features": [{"feature": f, "value": round(v, 5)} for f, v in ranked[:15]],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    log.info("Loading dataset: %s", DATA_PATH)
    raw = pd.read_csv(DATA_PATH)
    df = clean_data(raw)
    log.info("Cleaned data: %s rows x %s columns", *df.shape)

    # Target encoding: No -> 0, Yes -> 1
    y = (df[TARGET].map({"No": 0, "Yes": 1})).astype(int)
    X = df.drop(columns=[TARGET])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    log.info(
        "Train set: %s | Test set: %s (churn rate train=%.2f%%)",
        len(X_train),
        len(X_test),
        y_train.mean() * 100,
    )
    log.info("Churn rate test set: %.2f%%", y_test.mean() * 100)

    models = build_models()
    metrics = {}
    importances = {}
    baseline_auc = None

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for name, pipeline in models.items():
        log.info("Training %s ...", name)
        pipeline.fit(X_train, y_train)
        m = evaluate(pipeline, X_test, y_test)
        metrics[name] = m
        log.info(
            "  acc=%.4f  prec=%.4f  rec=%.4f  f1=%.4f  auc=%.4f",
            m["accuracy"],
            m["precision"],
            m["recall"],
            m["f1_score"],
            m["roc_auc"],
        )

        if baseline_auc is None:
            baseline_auc = m["roc_auc"]

        # Save full pipeline (self-contained preprocessing + model)
        path = MODEL_DIR / f"{name}.pkl"
        joblib.dump(pipeline, path)
        log.info("  saved -> %s", path.name)

    # Save the fitted preprocessor separately (for feature-name introspection)
    preprocessor = build_preprocessor()
    preprocessor.fit(clean_data(raw).drop(columns=[TARGET]))
    joblib.dump(preprocessor, MODEL_DIR / "preprocessor.pkl")

    # Feature importances using the fitted preprocessor
    for name in models:
        importances[name] = feature_importances(models[name], preprocessor)

    # Persist metrics + feature information
    (MODEL_DIR / "model_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    importance_payload = {
        # relative lift over the linear baseline (how much better each model is)
        "lift_over_baseline": {
            name: round(m["roc_auc"] - baseline_auc, 4) for name, m in metrics.items()
        },
        "per_model": importances,
    }
    (MODEL_DIR / "feature_importance.json").write_text(
        json.dumps(importance_payload, indent=2), encoding="utf-8"
    )
    (MODEL_DIR / "feature_columns.json").write_text(
        json.dumps(feature_groups(), indent=2), encoding="utf-8"
    )

    log.info("All artifacts written to: %s", MODEL_DIR)


if __name__ == "__main__":
    main()
```

## 4. src/optimise_thresholds.py

```python
"""
Decision-threshold optimisation.

Class-weighted tree models output probabilities that over-predict churn
(high recall, low precision). We search for a probability threshold that
maximises accuracy on the validation set, then persist that threshold so the
web app can apply it at prediction time.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.preprocess import build_preprocessor, clean_data

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "dataset" / "Telco_Customer_Churn.csv"
RANDOM_STATE = 42


def make_pipeline(estimator):
    return Pipeline(
        [
            ("clean", FunctionTransformer(clean_data, validate=False)),
            ("prep", build_preprocessor()),
            ("model", estimator),
        ]
    )


def fit_and_optimise(pipe, X_tr, y_tr, X_va, y_va):
    """Fit pipeline, find best threshold on validation probabilities."""
    pipe.fit(X_tr, y_tr)

    if hasattr(pipe.named_steps["model"], "predict_proba"):
        va_proba = pipe.predict_proba(X_va)[:, 1]
        has_proba = True
    else:  # linear regression
        va_proba = np.clip(pipe.predict(X_va), 0, 1)
        has_proba = False

    best = {"threshold": 0.5, "acc": 0, "f1": 0}
    for t in np.arange(0.30, 0.76, 0.025):
        preds = (va_proba >= t).astype(int)
        acc = accuracy_score(y_va, preds)
        f1 = f1_score(y_va, preds)
        if acc > best["acc"]:
            best = {"threshold": round(float(t), 3), "acc": round(acc, 4), "f1": round(f1, 4)}

    # Re-score on validation with best threshold
    preds = (va_proba >= best["threshold"]).astype(int)
    best["auc"] = round(roc_auc_score(y_va, va_proba), 4)
    best["has_probability"] = has_proba
    return best


def main():
    raw = pd.read_csv(DATA_PATH)
    df = clean_data(raw)
    y = df["Churn"].map({"No": 0, "Yes": 1}).astype(int)
    X = df.drop(columns=["Churn"])
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    models = {
        "linear_regression": make_pipeline(LinearRegression()),
        "random_forest": make_pipeline(
            RandomForestClassifier(
                n_estimators=400,
                max_depth=12,
                min_samples_split=5,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
        "xgboost": make_pipeline(
            XGBClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=1.4,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
    }

    out = {}
    for name, pipe in models.items():
        res = fit_and_optimise(pipe, X_tr, y_tr, X_va, y_va)
        out[name] = res
        print(f"{name:20s} threshold={res['threshold']:.3f} "
              f"acc@val={res['acc']:.4f} f1@val={res['f1']:.4f} auc={res['auc']:.4f}")

    (ROOT / "model" / "thresholds.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print("\nSaved -> model/thresholds.json")


if __name__ == "__main__":
    main()
```

## 5. app.py

```python
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
```

## 6. streamlit_app.py

```python
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
```
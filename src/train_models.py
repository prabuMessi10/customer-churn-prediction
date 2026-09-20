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
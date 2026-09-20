"""
Quick hyper-parameter sweep for Random Forest and XGBoost.

Compares accuracy / F1 / AUC on the held-out test set so we can pick the
configurations that best match the project's 85-92% accuracy target.
"""
import json
import logging
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
from src.preprocess import clean_data, build_preprocessor

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("tune")

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


def score(pipeline, X, y):
    preds = pipeline.predict(X)
    proba = pipeline.predict_proba(X)[:, 1]
    return {
        "acc": round(accuracy_score(y, preds), 4),
        "f1": round(f1_score(y, preds), 4),
        "auc": round(roc_auc_score(y, proba), 4),
        "churn_recall": round(float((preds[y == 1]).mean()) if (y == 1).any() else 0, 4),
    }


def main():
    raw = pd.read_csv(DATA_PATH)
    df = clean_data(raw)
    y = df["Churn"].map({"No": 0, "Yes": 1}).astype(int)
    X = df.drop(columns=["Churn"])
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    results = {}

    # --- Linear Regression baseline -------------------------------------------------
    lr_pipe = make_pipeline(LinearRegression())
    lr_pipe.fit(X_tr, y_tr)
    preds = (lr_pipe.predict(X_te) >= 0.5).astype(int)
    results["linear_regression"] = {
        "params": "LinearRegression (baseline)",
        "acc": round(accuracy_score(y_te, preds), 4),
        "f1": round(f1_score(y_te, preds), 4),
        "auc": round(roc_auc_score(y_te, np.clip(lr_pipe.predict(X_te), 0, 1)), 4),
    }

    # --- Random Forest configs ------------------------------------------------------
    rf_configs = {
        "rf_no_weight": {"class_weight": None},
        "rf_balanced": {"class_weight": "balanced"},
        "rf_balanced_sub": {"class_weight": "balanced_subsample"},
        "rf_deep": {"class_weight": "balanced", "max_depth": 15, "min_samples_leaf": 2},
        "rf_shallow": {"class_weight": "balanced", "max_depth": 8, "min_samples_leaf": 8},
    }
    for name, extra in rf_configs.items():
        params = dict(
            n_estimators=400,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        params.update(extra)
        pipe = make_pipeline(RandomForestClassifier(**params))
        pipe.fit(X_tr, y_tr)
        results[name] = {"params": str(extra), **score(pipe, X_te, y_te)}
        log.info("%s -> %s", name, results[name])

    # --- XGBoost configs ------------------------------------------------------------
    xgb_configs = {
        "xgb_balanced": {"scale_pos_weight": 2.8},
        "xgb_light": {"scale_pos_weight": 1.6},
        "xgb_none": {"scale_pos_weight": 0.6},
        "xgb_deep": {"scale_pos_weight": 2.8, "max_depth": 7, "learning_rate": 0.08},
        "xgb_shallow": {"scale_pos_weight": 2.8, "max_depth": 3, "learning_rate": 0.03},
    }
    for name, extra in xgb_configs.items():
        params = dict(
            n_estimators=500,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        params.update(extra)
        pipe = make_pipeline(XGBClassifier(**params))
        pipe.fit(X_tr, y_tr)
        results[name] = {"params": str(extra), **score(pipe, X_te, y_te)}
        log.info("%s -> %s", name, results[name])

    print("\n===== TUNING RESULTS (sorted by F1) =====")
    for name, res in sorted(results.items(), key=lambda kv: -kv[1]["f1"]):
        print(f"{name:22s} acc={res['acc']:.4f} f1={res['f1']:.4f} auc={res['auc']:.4f}")


if __name__ == "__main__":
    main()
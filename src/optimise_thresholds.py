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
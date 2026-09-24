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
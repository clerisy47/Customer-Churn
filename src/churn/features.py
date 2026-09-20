"""Shared preprocessing pipeline for Telco churn models."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]


def infer_categorical_features(X: pd.DataFrame) -> list[str]:
    """Categorical columns = all non-numeric feature columns."""
    return [c for c in X.columns if c not in NUMERIC_FEATURES]


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """ColumnTransformer: scale numerics, one-hot encode categoricals."""
    categorical = infer_categorical_features(X)
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ],
        remainder="drop",
    )


def build_model_pipeline(estimator: Any, X: pd.DataFrame) -> Pipeline:
    """Full sklearn Pipeline: preprocessor + classifier."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X)),
            ("classifier", estimator),
        ]
    )

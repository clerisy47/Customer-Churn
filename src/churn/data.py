"""Load and clean the Telco Customer Churn dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from churn.config import DATA_PATH, RANDOM_STATE, TEST_SIZE

TARGET_COL = "Churn"
ID_COL = "customerID"


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Load the raw CSV from disk."""
    csv_path = path or DATA_PATH
    return pd.read_csv(csv_path)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop ID, coerce TotalCharges, encode Churn as 0/1, drop residual NaNs."""
    out = df.copy()
    if ID_COL in out.columns:
        out = out.drop(columns=[ID_COL])

    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
    out = out.dropna(subset=["TotalCharges"]).reset_index(drop=True)

    if out[TARGET_COL].dtype == object:
        out[TARGET_COL] = out[TARGET_COL].map({"Yes": 1, "No": 0})

    out[TARGET_COL] = out[TARGET_COL].astype(int)
    return out


def load_clean(path: Path | None = None) -> pd.DataFrame:
    """Load and clean the full dataset."""
    return clean_dataframe(load_raw(path))


def feature_target_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split features and target."""
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y


def train_test_frames(
    path: Path | None = None,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split for modeling."""
    df = load_clean(path)
    X, y = feature_target_split(df)
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

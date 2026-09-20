"""Shared paths and MLflow configuration."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "dataset.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

EXPERIMENT_NAME = "telco-churn"
MONITOR_EXPERIMENT_NAME = "churn-monitoring"
REGISTERED_MODEL_NAME = "telco-churn-classifier"

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", MLRUNS_DIR.as_uri())

RANDOM_STATE = 42
TEST_SIZE = 0.2
REFERENCE_FRAC = 0.7

# Drift thresholds for monitor summary / Airflow
MONTHLY_CHARGES_MEAN_DIFF_THRESHOLD = 5.0
INJECTED_DRIFT_FEATURES = ("MonthlyCharges", "tenure", "Contract")

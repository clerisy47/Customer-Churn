"""Airflow DAG: weekly drift check; retrain when significant drift is detected."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator

# Project root: .../Customer-Churn (parent of airflow/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
DRIFT_SUMMARY = REPORTS_DIR / "drift_summary.json"
UV = os.environ.get("UV_BIN", "uv")
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", (PROJECT_ROOT / "mlruns").as_uri()
)

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
}


def _decide_retrain(**_context) -> str:
    if not DRIFT_SUMMARY.exists():
        print("No drift_summary.json found — skipping retrain.")
        return "skip_retrain"
    summary = json.loads(DRIFT_SUMMARY.read_text())
    if summary.get("significant_drift"):
        print("RETRAIN RECOMMENDED — significant drift detected.")
        print(json.dumps(summary, indent=2))
        return "trigger_retrain"
    print("NO RETRAIN NEEDED — drift within thresholds.")
    return "skip_retrain"


def _skip_retrain(**_context) -> None:
    print("Drift check complete; no retrain triggered.")


with DAG(
    dag_id="telco_churn_drift_monitor",
    description="Weekly Evidently drift check; retrain on significant drift",
    default_args=default_args,
    schedule="@weekly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["churn", "drift", "mlops"],
) as dag:
    run_drift_check = BashOperator(
        task_id="run_drift_check",
        bash_command=(
            f'cd "{PROJECT_ROOT}" && '
            f'MLFLOW_TRACKING_URI="{MLFLOW_TRACKING_URI}" '
            f"{UV} run python -m churn.monitor"
        ),
        env={
            **os.environ,
            "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI,
            "PATH": os.environ.get("PATH", ""),
        },
    )

    evaluate_drift = BranchPythonOperator(
        task_id="evaluate_drift",
        python_callable=_decide_retrain,
    )

    trigger_retrain = BashOperator(
        task_id="trigger_retrain",
        bash_command=(
            f'cd "{PROJECT_ROOT}" && '
            f'MLFLOW_TRACKING_URI="{MLFLOW_TRACKING_URI}" '
            f"{UV} run python -m churn.train"
        ),
        env={
            **os.environ,
            "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI,
            "PATH": os.environ.get("PATH", ""),
        },
    )

    skip_retrain = PythonOperator(
        task_id="skip_retrain",
        python_callable=_skip_retrain,
    )

    run_drift_check >> evaluate_drift >> [trigger_retrain, skip_retrain]

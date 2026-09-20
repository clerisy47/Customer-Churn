# Local Airflow setup (Telco churn drift DAG)

The DAG [`dags/churn_drift_dag.py`](dags/churn_drift_dag.py) runs weekly:

1. `run_drift_check` — `uv run python -m churn.monitor`
2. `evaluate_drift` — reads `reports/drift_summary.json`
3. On significant drift → `trigger_retrain` (`uv run python -m churn.train`); otherwise `skip_retrain`

## Install Airflow (optional extra)

From the project root:

```bash
uv sync --extra airflow
```

## Configure AIRFLOW_HOME

```bash
export AIRFLOW_HOME="$(pwd)/airflow"
export AIRFLOW__CORE__DAGS_FOLDER="$(pwd)/airflow/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES=False
export MLFLOW_TRACKING_URI="$(pwd)/mlruns"
# Ensure `uv` is on PATH (or set UV_BIN to the absolute uv binary)
```

## Initialize and start

```bash
cd /path/to/Customer-Churn
uv run airflow db migrate
uv run airflow users create \
  --username admin --password admin \
  --firstname Admin --lastname User \
  --role Admin --email admin@example.com

# All-in-one (webserver + scheduler)
uv run airflow standalone
```

Open http://localhost:8080 and trigger `telco_churn_drift_monitor` manually for a demo.

## Notes

- The DAG resolves the project root as the parent of `airflow/`, so it works when `AIRFLOW_HOME` is this `airflow/` directory.
- Core ML work (`uv sync` without extras) does **not** require Airflow; install the extra only when running the DAG.

# Telco Customer Churn — Production MLOps Pipeline

End-to-end MLOps workflow on the IBM Telco Customer Churn dataset (~7,000 customers):

**data → training → MLflow tracking → model registry → FastAPI serving → Evidently monitoring → Airflow-triggered retraining**

## Stack

| Concern | Tool |
|--------|------|
| Environment | [uv](https://github.com/astral-sh/uv) (`pyproject.toml` + `uv.lock`) |
| Tracking / registry | MLflow 2.x (local `./mlruns`) |
| Models | Logistic Regression, Random Forest, Gradient Boosting (scikit-learn) |
| Serving | FastAPI (`models:/telco-churn-classifier/Production`) |
| Drift | Evidently AI (HTML reports + custom metric) |
| Orchestration | Apache Airflow (optional extra) |

## Setup

Requires Python 3.11–3.12 (pinned via `.python-version`).

```bash
uv sync
```

Airflow (bonus orchestration) is optional:

```bash
uv sync --extra airflow
```

## 1. Train, track, and register

```bash
./scripts/run_train.sh
# or: uv run python -m churn.train
```

This script:

1. Trains **3** models with distinct hyperparameters:
   - Logistic Regression (`C=0.1`)
   - Random Forest (`n_estimators=100`, `max_depth=5`)
   - Gradient Boosting (`n_estimators=200`, `learning_rate=0.05`)
2. Logs params, metrics (**accuracy, precision, recall, F1, ROC-AUC**), model artifact, confusion matrix, and ROC curve to MLflow experiment `telco-churn`.
3. Selects the best run by **ROC-AUC** (tie-break: F1) — appropriate for class imbalance (~26.5% churn).
4. Registers `telco-churn-classifier` and transitions **Staging → Production**.
5. Writes a side-by-side table to [`reports/run_comparison.csv`](reports/run_comparison.csv).

Compare runs in the UI:

```bash
uv run mlflow ui --backend-store-uri ./mlruns --port 5000
```

Open http://localhost:5000 → experiment **telco-churn**.

## 2. Serve the Production model

```bash
./scripts/run_serve.sh
# or: uv run uvicorn churn.serve:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl -s http://localhost:8000/health | python -m json.tool
```

Example prediction:

```bash
curl -s -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "customers": [{
      "gender": "Female",
      "SeniorCitizen": 0,
      "Partner": "Yes",
      "Dependents": "No",
      "tenure": 1,
      "PhoneService": "No",
      "MultipleLines": "No phone service",
      "InternetService": "DSL",
      "OnlineSecurity": "No",
      "OnlineBackup": "Yes",
      "DeviceProtection": "No",
      "TechSupport": "No",
      "StreamingTV": "No",
      "StreamingMovies": "No",
      "Contract": "Month-to-month",
      "PaperlessBilling": "Yes",
      "PaymentMethod": "Electronic check",
      "MonthlyCharges": 29.85,
      "TotalCharges": 29.85
    }]
  }' | python -m json.tool
```

## 3. Drift monitoring (Evidently)

```bash
./scripts/run_monitor.sh
# or: uv run python -m churn.monitor
```

What it does:

1. Splits cleaned data into **70% reference** / **30% current**.
2. Injects synthetic drift into current only:
   - Noise on `MonthlyCharges` and `tenure`
   - Skews `Contract` toward `Month-to-month`
   - Flips ~20% of `Churn` labels
3. Builds Evidently **data drift** and **target drift** HTML reports.
4. Computes a **custom Evidently metric** `MonthlyChargesMeanAbsDiff` (absolute mean difference vs threshold 5.0).
5. Logs HTML + metrics + interpretation to MLflow experiment `churn-monitoring`.
6. Writes `reports/drift_summary.json` (`significant_drift`, recommendation) for Airflow.

Artifacts:

- `reports/drift_report.html`
- `reports/target_drift_report.html`
- `reports/interpretation.md`
- `reports/drift_summary.json`

### Interpretation (typical run)

Engineered drift in `MonthlyCharges`, `tenure`, and `Contract` is flagged by Evidently; target drift on `Churn` is flagged when label flips are large enough. The custom mean-diff metric exceeds the threshold (~24 vs 5). In production this would mean the live population no longer matches training — investigate pipelines and **retrain** before trusting Production scores.

## 4. Airflow orchestration (bonus)

See [`airflow/README.md`](airflow/README.md). Weekly DAG `telco_churn_drift_monitor`:

`run_drift_check` → `evaluate_drift` → `trigger_retrain` **or** `skip_retrain`

## Project layout

```
src/churn/          # data, features, train, serve, monitor
airflow/dags/       # weekly drift DAG
scripts/            # thin wrappers
reports/            # comparison CSV, drift HTML, summaries
mlruns/             # local MLflow store (gitignored)
dataset.csv         # Telco Customer Churn
```

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `MLFLOW_TRACKING_URI` | `./mlruns` (file URI) | Tracking / registry location |
| `UV_BIN` | `uv` | Used by the Airflow DAG |
| `AIRFLOW_HOME` | — | Set to `./airflow` for local DAG runs |

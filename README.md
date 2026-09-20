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

---

## Documentation Requirements

### a. Environment & Reproducibility (uv)

**What uv solves for this project specifically**

This pipeline mixes packages that routinely break across environments if versions are free-floating:

- **Python range** — MLflow 2.x + Evidently + Airflow need a consistent interpreter; we pin **3.11** via `.python-version` and `requires-python = ">=3.11,<3.13"`.
- **MLflow major line** — `mlflow>=2.14,<3` avoids MLflow 3 API/registry changes that would invalidate our `sklearn.log_model` + stage-transition flow.
- **Airflow as an optional extra** — `apache-airflow>=2.9,<2.11` is heavy and conflict-prone; core train/serve/monitor work without it. `uv sync --extra airflow` installs that stack only when needed.
- **Lockfile** — `uv.lock` freezes transitive versions (scikit-learn, Evidently, FastAPI, etc.) so two clones get the same resolved graph, not “whatever pip resolved today.”

**One-command path from a clean clone**

```bash
git clone <repo-url> Customer-Churn && cd Customer-Churn
uv sync
./scripts/run_train.sh      # trains ≥3 models, logs to MLflow, registers Production winner
./scripts/run_monitor.sh    # Evidently drift report → reports/drift_summary.json
# optional serving: ./scripts/run_serve.sh
# optional orchestration: uv sync --extra airflow  (see airflow/README.md)
```

Confirmed: after `uv sync`, `uv run python -m churn.train` and `uv run python -m churn.monitor` reproduce training metrics, registry promotion, and drift artifacts without manual `pip install` or ad-hoc version pins.

### b. Experiment Tracking Strategy (MLflow)

**What we varied and why**

Three runs in experiment `telco-churn`, each a different model family with distinct hyperparameters (not the same estimator retuned three times):

| Run name | Family | Key hyperparameters |
|----------|--------|---------------------|
| `logistic_regression_c0.1` | Logistic Regression | `C=0.1`, `max_iter=1000`, `solver=lbfgs` |
| `random_forest_shallow` | Random Forest | `n_estimators=100`, `max_depth=5`, `min_samples_leaf=5` |
| `gradient_boosting_deeper` | Gradient Boosting | `n_estimators=200`, `max_depth=3`, `learning_rate=0.05` |

Rationale: compare a linear baseline vs. bagging vs. boosting on the same train/test split and feature pipeline, so differences reflect inductive bias and capacity—not data leakage or different preprocessing.

**What we measured**

Per run: **accuracy, precision, recall, F1, ROC-AUC**, plus confusion matrix and ROC curve artifacts.

Churn is imbalanced (~26.5% positive). Accuracy alone overstates “good” models that mostly predict non-churn; we therefore rank by **ROC-AUC** (tie-break: F1) for registry promotion.

**Side-by-side comparison (actual MLflow / `reports/run_comparison.csv`)**

| Run | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-----|----------|-----------|--------|-----|---------|
| `gradient_boosting_deeper` | 0.7925 | 0.6306 | 0.5294 | 0.5756 | **0.8392** |
| `logistic_regression_c0.1` | **0.8053** | **0.6543** | **0.5668** | **0.6074** | 0.8352 |
| `random_forest_shallow` | 0.7854 | 0.6488 | 0.4198 | 0.5097 | 0.8341 |

Full CSV: [`reports/run_comparison.csv`](reports/run_comparison.csv). UI: `uv run mlflow ui --backend-store-uri ./mlruns --port 5000` → experiment **telco-churn**.

**Which model we registered and why (backed by the table)**

Only **`gradient_boosting_deeper`** was registered as `telco-churn-classifier` and promoted Staging → Production (`reports/training_summary.json`: ROC-AUC **0.8392**, F1 **0.5756**).

Decision drivers from the numbers above:

- Gradient Boosting had the **highest ROC-AUC (0.8392)** vs. Logistic Regression (0.8352) and Random Forest (0.8341).
- Logistic Regression had **higher accuracy (0.8053 vs 0.7925)** and **higher F1 (0.6074 vs 0.5756)** — we did **not** pick it, because on this imbalanced target accuracy is misleading and we prioritize ranking quality for thresholded retention actions.
- Random Forest’s ROC-AUC was close (0.8341) but **recall collapsed to 0.4198** (vs 0.5294 for GB and 0.5668 for LR), so it misses too many churners for a retention use case.

**Trade-off accepted:** ~1.3 pp lower accuracy and ~3.2 pp lower F1 versus Logistic Regression, in exchange for the best discrimination (ROC-AUC) for scoring and ranking at-risk customers. LR remains a strong, simpler baseline; RF was not competitive on recall.

### c. Monitoring & Drift Strategy (Evidently AI)

**Reference vs. current in this setup**

- **Reference** — random **70%** of the cleaned Telco table: the baseline “training-time” feature/label distribution.
- **Current** — remaining **30%**, then deliberately perturbed to simulate production shift (noise on `MonthlyCharges` / `tenure`, Contract skewed toward `Month-to-month`, ~20% of `Churn` labels flipped).

**Metrics monitored**

- Evidently **data drift** and **target drift** (HTML reports).
- Custom Evidently metric **`MonthlyChargesMeanAbsDiff`** vs threshold **5.0**.
- Segment check: absolute churn-rate shift within `Contract == Month-to-month`.
- Aggregated flag `significant_drift` written to `reports/drift_summary.json` for Airflow.

**What a typical report showed** (`reports/drift_summary.json` / `reports/interpretation.md`)

| Signal | Value |
|--------|-------|
| Dataset drift flagged | `true` |
| Drifted columns | `Churn`, `Contract`, `MonthlyCharges`, `tenure` |
| Injected features that were flagged | `MonthlyCharges`, `tenure`, `Contract` |
| `MonthlyCharges` mean abs diff | **24.41** (threshold 5.0) |
| Month-to-month churn-rate abs shift | **0.0275** |
| Recommendation | **RETRAIN RECOMMENDED** |

Artifacts: `reports/drift_report.html`, `reports/target_drift_report.html`, `reports/interpretation.md`.

**Action if drift crosses a concerning threshold**

If `significant_drift` is true (dataset drift and/or custom mean-diff over threshold):

1. Treat Production scores as unreliable until investigated (pipelines / population shift).
2. Automated path: Airflow branches to **retrain** (`python -m churn.train`), which re-logs runs, re-selects by ROC-AUC, and re-promotes Production.
3. Human follow-up: review HTML reports, confirm the shift is real (not a broken feed), then validate the new Production model before trusting live predictions.

### d. Orchestration

DAG id: **`telco_churn_drift_monitor`** ([`airflow/dags/churn_drift_dag.py`](airflow/dags/churn_drift_dag.py)).

| Piece | Detail |
|-------|--------|
| **Schedule** | `@weekly` (`catchup=False`) |
| **Trigger condition** | After `run_drift_check`, `evaluate_drift` reads `reports/drift_summary.json` and branches on `significant_drift == true` |
| **On positive drift / degradation** | `trigger_retrain` → `uv run python -m churn.train` (new MLflow runs + registry promotion) |
| **Otherwise** | `skip_retrain` (log only; Production model unchanged) |

```
run_drift_check  →  evaluate_drift  →  trigger_retrain
                                  ↘  skip_retrain
```

Setup: [`airflow/README.md`](airflow/README.md).

---

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

See **§c** above for reference/current semantics, metrics, and response policy.

## 4. Airflow orchestration (bonus)

See **§d** and [`airflow/README.md`](airflow/README.md).

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

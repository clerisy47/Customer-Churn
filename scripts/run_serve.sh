#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-$ROOT/mlruns}"
uv run uvicorn churn.serve:app --host 0.0.0.0 --port 8000

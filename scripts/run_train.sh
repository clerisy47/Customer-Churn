#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-$ROOT/mlruns}"
uv run python -m churn.train

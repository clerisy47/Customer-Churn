"""Evidently drift monitoring with synthetic drift injection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from evidently import Dataset, DataDefinition, Report
from evidently.metrics import DriftedColumnsCount, MeanValue, ValueDrift
from evidently.presets import DataDriftPreset
from sklearn.model_selection import train_test_split

from churn.config import (
    INJECTED_DRIFT_FEATURES,
    MONITOR_EXPERIMENT_NAME,
    MONTHLY_CHARGES_MEAN_DIFF_THRESHOLD,
    PROJECT_ROOT,
    RANDOM_STATE,
    REFERENCE_FRAC,
    REPORTS_DIR,
    TRACKING_URI,
)
from churn.data import TARGET_COL, load_clean
from churn.evidently_metrics import MonthlyChargesMeanAbsDiff
from churn.features import NUMERIC_FEATURES


def _inject_drift(current: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Deliberately perturb current data so drift reports have something to flag."""
    out = current.copy()

    out["MonthlyCharges"] = out["MonthlyCharges"] + rng.normal(
        loc=25.0, scale=8.0, size=len(out)
    )
    out["MonthlyCharges"] = out["MonthlyCharges"].clip(lower=0)
    out["tenure"] = out["tenure"] + rng.integers(low=6, high=18, size=len(out))
    out["tenure"] = out["tenure"].clip(lower=0, upper=72)

    n_force = max(1, int(0.45 * len(out)))
    force_idx = rng.choice(out.index.to_numpy(), size=n_force, replace=False)
    out.loc[force_idx, "Contract"] = "Month-to-month"

    # Flip a larger share of labels so target drift is clearly detectable
    n_flip = max(1, int(0.20 * len(out)))
    flip_idx = rng.choice(out.index.to_numpy(), size=n_flip, replace=False)
    out.loc[flip_idx, TARGET_COL] = 1 - out.loc[flip_idx, TARGET_COL]

    return out


def _data_definition(include_target: bool = False) -> DataDefinition:
    categorical = [
        c
        for c in [
            "gender",
            "Partner",
            "Dependents",
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
        ]
    ]
    numerical = list(NUMERIC_FEATURES)
    if include_target:
        numerical = numerical + [TARGET_COL]
    return DataDefinition(numerical_columns=numerical, categorical_columns=categorical)


def _to_dataset(df: pd.DataFrame, include_target: bool = False) -> Dataset:
    frame = df if include_target else df.drop(columns=[TARGET_COL])
    return Dataset.from_pandas(frame, data_definition=_data_definition(include_target))


def _parse_drifted_columns(metrics_json: list[dict[str, Any]]) -> tuple[list[str], bool]:
    drifted: list[str] = []
    dataset_drift = False
    for metric in metrics_json:
        name = metric.get("metric_name", "")
        value = metric.get("value")
        config = metric.get("config", {})
        if name.startswith("DriftedColumnsCount"):
            if isinstance(value, dict) and value.get("count", 0) > 0:
                dataset_drift = True
            continue
        if name.startswith("ValueDrift") and "column" in config:
            threshold = float(config.get("threshold", 0.05))
            method = str(config.get("method", "")).lower()
            try:
                score = float(value)
            except (TypeError, ValueError):
                continue
            # p-value methods: drift when p < threshold
            # distance methods (Wasserstein, Jensen-Shannon, etc.): drift when score > threshold
            is_pvalue = "p_value" in method or "p-value" in method or "pvalue" in method
            drifted_flag = score < threshold if is_pvalue else score > threshold
            if drifted_flag:
                drifted.append(config["column"])
    return sorted(set(drifted)), dataset_drift or bool(drifted)


def _contract_segment_churn_shift(
    reference: pd.DataFrame, current: pd.DataFrame
) -> float:
    ref_seg = reference[reference["Contract"] == "Month-to-month"]
    cur_seg = current[current["Contract"] == "Month-to-month"]
    ref_rate = float(ref_seg[TARGET_COL].mean()) if len(ref_seg) else 0.0
    cur_rate = float(cur_seg[TARGET_COL].mean()) if len(cur_seg) else 0.0
    return float(abs(cur_rate - ref_rate))


def _build_reports(
    reference: pd.DataFrame, current: pd.DataFrame
) -> tuple[Path, Path, dict[str, Any], float]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    drift_html = REPORTS_DIR / "drift_report.html"
    target_html = REPORTS_DIR / "target_drift_report.html"

    ref_feat = _to_dataset(reference, include_target=False)
    cur_feat = _to_dataset(current, include_target=False)

    data_report = Report(
        [
            DataDriftPreset(),
            DriftedColumnsCount(),
            MeanValue(column="MonthlyCharges"),
            MonthlyChargesMeanAbsDiff(
                column="MonthlyCharges",
                threshold=MONTHLY_CHARGES_MEAN_DIFF_THRESHOLD,
            ),
        ]
    )
    data_snap = data_report.run(cur_feat, ref_feat)
    data_snap.save_html(str(drift_html))
    data_metrics = json.loads(data_snap.json())["metrics"]

    drifted_columns, dataset_drift = _parse_drifted_columns(data_metrics)

    # Extract custom metric value from report JSON
    custom_mean_diff = None
    for metric in data_metrics:
        if "MonthlyChargesMeanAbsDiff" in metric.get("metric_name", "") or (
            metric.get("config", {}).get("type", "").endswith("MonthlyChargesMeanAbsDiff")
        ):
            try:
                custom_mean_diff = float(metric["value"])
            except (TypeError, ValueError, KeyError):
                pass
            break
    if custom_mean_diff is None:
        custom_mean_diff = float(
            abs(current["MonthlyCharges"].mean() - reference["MonthlyCharges"].mean())
        )

    # Target drift report
    ref_full = _to_dataset(reference, include_target=True)
    cur_full = _to_dataset(current, include_target=True)
    target_report = Report([ValueDrift(column=TARGET_COL, threshold=0.05)])
    target_snap = target_report.run(cur_full, ref_full)
    target_snap.save_html(str(target_html))
    target_metrics = json.loads(target_snap.json())["metrics"]
    target_drifted, _ = _parse_drifted_columns(target_metrics)
    drifted_columns = sorted(set(drifted_columns) | set(target_drifted))

    return (
        drift_html,
        target_html,
        {
            "drifted_columns": drifted_columns,
            "dataset_drift": bool(dataset_drift or drifted_columns),
        },
        custom_mean_diff,
    )


def run_monitoring() -> dict[str, Any]:
    """Split data, inject drift, run Evidently, log to MLflow, write summary JSON."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_clean()
    reference, current = train_test_split(
        df,
        test_size=1.0 - REFERENCE_FRAC,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COL],
    )
    reference = reference.reset_index(drop=True)
    current = current.reset_index(drop=True)

    rng = np.random.default_rng(RANDOM_STATE)
    current_drifted = _inject_drift(current, rng)

    drift_html, target_html, drift_info, mean_diff = _build_reports(
        reference, current_drifted
    )
    segment_shift = _contract_segment_churn_shift(reference, current_drifted)

    injected_flagged = [
        c for c in INJECTED_DRIFT_FEATURES if c in drift_info["drifted_columns"]
    ]
    custom_triggered = mean_diff >= MONTHLY_CHARGES_MEAN_DIFF_THRESHOLD
    significant_drift = bool(
        drift_info["dataset_drift"]
        or injected_flagged
        or custom_triggered
        or TARGET_COL in drift_info["drifted_columns"]
    )

    interpretation = f"""# Drift Monitoring Interpretation

## Setup
- Reference: random {REFERENCE_FRAC:.0%} of Telco churn data (training-time distribution).
- Current: remaining {(1 - REFERENCE_FRAC):.0%}, with **synthetic drift** injected.

## Engineered perturbations
- `MonthlyCharges`: shifted by ~N(25, 8) noise per row.
- `tenure`: increased by a random integer offset in [6, 18].
- `Contract`: ~45% of current rows forced to `Month-to-month`.
- `Churn`: ~20% of labels flipped to simulate concept/label drift.

## Custom metrics
- Absolute mean difference in `MonthlyCharges` (Evidently custom metric): **{mean_diff:.3f}**
  (threshold={MONTHLY_CHARGES_MEAN_DIFF_THRESHOLD}).
- Absolute churn-rate shift within `Contract == Month-to-month`: **{segment_shift:.4f}**.

## Detected drift
- Dataset drift flagged: **{drift_info['dataset_drift']}**
- Drifted columns reported: {drift_info['drifted_columns'] or 'none parsed'}
- Injected features that were flagged: {injected_flagged or 'none (see HTML for details)'}
- Significant drift (for Airflow): **{significant_drift}**

## Production implication
If this pattern appeared in production, feature and/or target distributions would no longer
match training data. Predictions (especially for monthly-charge-sensitive and month-to-month
segments) would be unreliable. Recommended action: investigate data pipelines, then **retrain**
and re-evaluate before promoting a new Production model.
"""
    interpretation_path = REPORTS_DIR / "interpretation.md"
    interpretation_path.write_text(interpretation)

    summary = {
        "significant_drift": significant_drift,
        "dataset_drift": drift_info["dataset_drift"],
        "drifted_columns": drift_info["drifted_columns"],
        "injected_features_flagged": injected_flagged,
        "custom_metrics": {
            "monthly_charges_mean_abs_diff": mean_diff,
            "monthly_charges_mean_diff_threshold": MONTHLY_CHARGES_MEAN_DIFF_THRESHOLD,
            "month_to_month_churn_rate_abs_shift": segment_shift,
        },
        "artifacts": {
            "drift_report_html": str(drift_html.relative_to(PROJECT_ROOT)),
            "target_drift_report_html": str(target_html.relative_to(PROJECT_ROOT)),
            "interpretation_md": str(interpretation_path.relative_to(PROJECT_ROOT)),
        },
        "recommendation": (
            "RETRAIN RECOMMENDED" if significant_drift else "NO RETRAIN NEEDED"
        ),
    }
    summary_path = REPORTS_DIR / "drift_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(MONITOR_EXPERIMENT_NAME)
    with mlflow.start_run(run_name="evidently_drift_check"):
        mlflow.log_metric("monthly_charges_mean_abs_diff", mean_diff)
        mlflow.log_metric("month_to_month_churn_rate_abs_shift", segment_shift)
        mlflow.log_metric("significant_drift", 1.0 if significant_drift else 0.0)
        mlflow.log_param("reference_frac", REFERENCE_FRAC)
        mlflow.log_param("injected_features", ",".join(INJECTED_DRIFT_FEATURES))
        for path in (drift_html, target_html, interpretation_path, summary_path):
            if path.exists():
                mlflow.log_artifact(str(path))

    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    run_monitoring()


if __name__ == "__main__":
    main()

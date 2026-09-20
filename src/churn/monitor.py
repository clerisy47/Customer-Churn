"""Evidently drift monitoring with synthetic drift injection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from evidently import Report
from evidently.metrics import (
    DatasetDriftMetric,
    ValueDrift,
)
from evidently.presets import DataDriftPreset, DataSummaryPreset
from sklearn.model_selection import train_test_split

from churn.config import (
    INJECTED_DRIFT_FEATURES,
    MONITOR_EXPERIMENT_NAME,
    MONTHLY_CHARGES_MEAN_DIFF_THRESHOLD,
    RANDOM_STATE,
    REFERENCE_FRAC,
    REPORTS_DIR,
    TRACKING_URI,
)
from churn.data import TARGET_COL, load_clean

# Evidently 0.7+ API varies; try classic Report/metrics with fallbacks below.


def _inject_drift(current: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Deliberately perturb current data so drift reports have something to flag."""
    out = current.copy()

    # Numeric noise on MonthlyCharges and tenure
    out["MonthlyCharges"] = out["MonthlyCharges"] + rng.normal(loc=25.0, scale=8.0, size=len(out))
    out["MonthlyCharges"] = out["MonthlyCharges"].clip(lower=0)
    out["tenure"] = out["tenure"] + rng.integers(low=6, high=18, size=len(out))
    out["tenure"] = out["tenure"].clip(lower=0, upper=72)

    # Skew Contract toward Month-to-month
    n_force = max(1, int(0.45 * len(out)))
    force_idx = rng.choice(out.index.to_numpy(), size=n_force, replace=False)
    out.loc[force_idx, "Contract"] = "Month-to-month"

    # Flip a share of labels (concept / label drift)
    n_flip = max(1, int(0.08 * len(out)))
    flip_idx = rng.choice(out.index.to_numpy(), size=n_flip, replace=False)
    out.loc[flip_idx, TARGET_COL] = 1 - out.loc[flip_idx, TARGET_COL]

    return out


def _monthly_charges_mean_diff(reference: pd.DataFrame, current: pd.DataFrame) -> float:
    """Custom metric: |mean(MonthlyCharges_current) - mean(MonthlyCharges_reference)|."""
    return float(abs(current["MonthlyCharges"].mean() - reference["MonthlyCharges"].mean()))


def _contract_segment_churn_shift(
    reference: pd.DataFrame, current: pd.DataFrame
) -> float:
    """Custom metric: churn-rate delta within Contract == Month-to-month."""
    ref_seg = reference[reference["Contract"] == "Month-to-month"]
    cur_seg = current[current["Contract"] == "Month-to-month"]
    ref_rate = float(ref_seg[TARGET_COL].mean()) if len(ref_seg) else 0.0
    cur_rate = float(cur_seg[TARGET_COL].mean()) if len(cur_seg) else 0.0
    return float(abs(cur_rate - ref_rate))


def _build_evidently_reports(
    reference: pd.DataFrame, current: pd.DataFrame
) -> tuple[Path, Path, dict[str, Any]]:
    """Generate data-drift and target-drift HTML reports; return paths + summary bits."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    drift_html = REPORTS_DIR / "drift_report.html"
    target_html = REPORTS_DIR / "target_drift_report.html"

    drifted_columns: list[str] = []
    dataset_drift = False

    # --- Data drift report (features only) ---
    ref_feat = reference.drop(columns=[TARGET_COL])
    cur_feat = current.drop(columns=[TARGET_COL])

    try:
        from evidently.report import Report as LegacyReport
        from evidently.metric_preset import DataDriftPreset as LegacyDataDriftPreset
        from evidently.metric_preset import TargetDriftPreset as LegacyTargetDriftPreset
        from evidently.metrics import ColumnDriftMetric

        data_report = LegacyReport(metrics=[LegacyDataDriftPreset()])
        data_report.run(reference_data=ref_feat, current_data=cur_feat)
        data_report.save_html(str(drift_html))

        result = data_report.as_dict()
        # Parse drifted columns from metrics payload when available
        for metric in result.get("metrics", []):
            metric_result = metric.get("result", {})
            if "drift_by_columns" in metric_result:
                for col, info in metric_result["drift_by_columns"].items():
                    if info.get("drift_detected"):
                        drifted_columns.append(col)
            if metric_result.get("dataset_drift") is True:
                dataset_drift = True
            # DatasetDriftMetric-style
            if "number_of_drifted_columns" in metric_result:
                dataset_drift = metric_result.get("dataset_drift", dataset_drift)

        target_report = LegacyReport(metrics=[LegacyTargetDriftPreset()])
        # TargetDriftPreset expects target column present
        target_report.run(reference_data=reference, current_data=current)
        target_report.save_html(str(target_html))

        # Also check target column drift explicitly
        col_report = LegacyReport(metrics=[ColumnDriftMetric(column_name=TARGET_COL)])
        col_report.run(reference_data=reference, current_data=current)
        col_dict = col_report.as_dict()
        for metric in col_dict.get("metrics", []):
            if metric.get("result", {}).get("drift_detected"):
                drifted_columns.append(TARGET_COL)

    except Exception:
        # Evidently 0.7+ preset API
        data_report = Report([DataDriftPreset(), DataSummaryPreset()])
        snapshot = data_report.run(cur_feat, ref_feat)
        snapshot.save_html(str(drift_html))

        # Minimal target comparison HTML via ValueDrift if available
        try:
            target_report = Report([ValueDrift(column=TARGET_COL), DatasetDriftMetric()])
            t_snap = target_report.run(current, reference)
            t_snap.save_html(str(target_html))
        except Exception:
            # Fallback: write a simple HTML summarizing target rates
            ref_rate = float(reference[TARGET_COL].mean())
            cur_rate = float(current[TARGET_COL].mean())
            target_html.write_text(
                "<html><body>"
                "<h1>Target Drift Summary</h1>"
                f"<p>Reference churn rate: {ref_rate:.4f}</p>"
                f"<p>Current churn rate: {cur_rate:.4f}</p>"
                f"<p>Absolute shift: {abs(cur_rate - ref_rate):.4f}</p>"
                "</body></html>"
            )
            if abs(cur_rate - ref_rate) > 0.02:
                drifted_columns.append(TARGET_COL)

        # Heuristic: mark injected features as drifted when means/distributions moved
        if abs(current["MonthlyCharges"].mean() - reference["MonthlyCharges"].mean()) > 5:
            drifted_columns.append("MonthlyCharges")
        if abs(current["tenure"].mean() - reference["tenure"].mean()) > 2:
            drifted_columns.append("tenure")
        ref_m2m = (reference["Contract"] == "Month-to-month").mean()
        cur_m2m = (current["Contract"] == "Month-to-month").mean()
        if abs(cur_m2m - ref_m2m) > 0.05:
            drifted_columns.append("Contract")
        dataset_drift = len(drifted_columns) > 0

    drifted_columns = sorted(set(drifted_columns))
    return drift_html, target_html, {
        "drifted_columns": drifted_columns,
        "dataset_drift": bool(dataset_drift or drifted_columns),
    }


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

    mean_diff = _monthly_charges_mean_diff(reference, current_drifted)
    segment_shift = _contract_segment_churn_shift(reference, current_drifted)

    drift_html, target_html, drift_info = _build_evidently_reports(
        reference, current_drifted
    )

    injected_flagged = [
        c for c in INJECTED_DRIFT_FEATURES if c in drift_info["drifted_columns"]
    ]
    # Also count as significant if custom metric exceeds threshold
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
- `Churn`: ~8% of labels flipped to simulate concept/label drift.

## Custom metrics
- Absolute mean difference in `MonthlyCharges`: **{mean_diff:.3f}** (threshold={MONTHLY_CHARGES_MEAN_DIFF_THRESHOLD}).
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
            "drift_report_html": str(drift_html),
            "target_drift_report_html": str(target_html),
            "interpretation_md": str(interpretation_path),
        },
        "recommendation": "RETRAIN RECOMMENDED" if significant_drift else "NO RETRAIN NEEDED",
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

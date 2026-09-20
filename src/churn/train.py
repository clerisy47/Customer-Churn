"""Train multiple churn models, log to MLflow, and promote the best to Production."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import seaborn as sns
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from churn.config import (
    EXPERIMENT_NAME,
    REGISTERED_MODEL_NAME,
    REPORTS_DIR,
    TRACKING_URI,
)
from churn.data import train_test_frames
from churn.features import build_model_pipeline


def _ensure_dirs() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _plot_confusion_matrix(y_true, y_pred, out_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_roc_curve(y_true, y_proba, out_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"ROC AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _evaluate(y_true, y_pred, y_proba) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def _model_configs() -> list[dict[str, Any]]:
    """Three runs with genuinely different hyperparameters / model families."""
    return [
        {
            "run_name": "logistic_regression_c0.1",
            "family": "logistic_regression",
            "params": {"C": 0.1, "max_iter": 1000, "solver": "lbfgs"},
            "estimator": LogisticRegression(
                C=0.1, max_iter=1000, solver="lbfgs", random_state=42
            ),
        },
        {
            "run_name": "random_forest_shallow",
            "family": "random_forest",
            "params": {
                "n_estimators": 100,
                "max_depth": 5,
                "min_samples_leaf": 5,
                "random_state": 42,
            },
            "estimator": RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_leaf=5,
                random_state=42,
            ),
        },
        {
            "run_name": "gradient_boosting_deeper",
            "family": "gradient_boosting",
            "params": {
                "n_estimators": 200,
                "max_depth": 3,
                "learning_rate": 0.05,
                "random_state": 42,
            },
            "estimator": GradientBoostingClassifier(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                random_state=42,
            ),
        },
    ]


def run_training() -> pd.DataFrame:
    """Train all configured models, log to MLflow, register and promote the best."""
    _ensure_dirs()
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = train_test_frames()
    rows: list[dict[str, Any]] = []

    for cfg in _model_configs():
        pipeline = build_model_pipeline(cfg["estimator"], X_train)
        with mlflow.start_run(run_name=cfg["run_name"]) as run:
            mlflow.set_tag("model_family", cfg["family"])
            mlflow.log_params(cfg["params"])
            mlflow.log_param("model_type", cfg["family"])

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            metrics = _evaluate(y_test, y_pred, y_proba)
            mlflow.log_metrics(metrics)

            cm_path = REPORTS_DIR / f"cm_{cfg['run_name']}.png"
            roc_path = REPORTS_DIR / f"roc_{cfg['run_name']}.png"
            _plot_confusion_matrix(y_test, y_pred, cm_path)
            _plot_roc_curve(y_test, y_proba, roc_path)
            mlflow.log_artifact(str(cm_path))
            mlflow.log_artifact(str(roc_path))

            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
                registered_model_name=None,
                input_example=X_test.head(3),
            )

            row = {
                "run_id": run.info.run_id,
                "run_name": cfg["run_name"],
                "family": cfg["family"],
                **cfg["params"],
                **metrics,
            }
            rows.append(row)
            print(
                f"[{cfg['run_name']}] "
                f"roc_auc={metrics['roc_auc']:.4f} f1={metrics['f1']:.4f}"
            )

    comparison = pd.DataFrame(rows)
    comparison = comparison.sort_values(
        by=["roc_auc", "f1"], ascending=False
    ).reset_index(drop=True)
    comparison_path = REPORTS_DIR / "run_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    print("\nRun comparison (sorted by roc_auc, then f1):")
    print(comparison[["run_name", "family", "accuracy", "precision", "recall", "f1", "roc_auc"]].to_string(index=False))

    best = comparison.iloc[0]
    best_run_id = best["run_id"]
    model_uri = f"runs:/{best_run_id}/model"

    client = MlflowClient()
    # Register (or create new version of) the winning model
    result = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
    version = result.version

    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=version,
        stage="Staging",
        archive_existing_versions=False,
    )
    print(f"Registered {REGISTERED_MODEL_NAME} v{version} → Staging")

    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"Promoted {REGISTERED_MODEL_NAME} v{version} → Production")

    summary = {
        "best_run_id": best_run_id,
        "best_run_name": best["run_name"],
        "best_roc_auc": float(best["roc_auc"]),
        "best_f1": float(best["f1"]),
        "registered_model": REGISTERED_MODEL_NAME,
        "model_version": version,
        "stages": ["Staging", "Production"],
    }
    summary_path = REPORTS_DIR / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nBest model: {best['run_name']} (roc_auc={best['roc_auc']:.4f})")
    return comparison


def main() -> None:
    run_training()


if __name__ == "__main__":
    main()

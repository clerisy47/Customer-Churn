"""FastAPI serving layer for the Production-registered churn model."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, ConfigDict, Field

from churn.config import REGISTERED_MODEL_NAME, TRACKING_URI

MODEL_URI = f"models:/{REGISTERED_MODEL_NAME}/Production"

_model = None
_model_version: str | None = None


class CustomerFeatures(BaseModel):
    """Raw Telco feature row (pre-encoding), matching training columns."""

    model_config = ConfigDict(extra="forbid")

    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


class PredictRequest(BaseModel):
    customers: list[CustomerFeatures]


class Prediction(BaseModel):
    churn_probability: float
    churn_prediction: int
    churn_label: str


class PredictResponse(BaseModel):
    model_name: str
    model_version: str | None
    predictions: list[Prediction]


def _load_production_model() -> tuple[Any, str | None]:
    mlflow.set_tracking_uri(TRACKING_URI)
    model = mlflow.sklearn.load_model(MODEL_URI)
    client = MlflowClient()
    versions = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["Production"])
    version = str(versions[0].version) if versions else None
    return model, version


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _model_version
    try:
        _model, _model_version = _load_production_model()
        print(f"Loaded {MODEL_URI} (version={_model_version})")
    except Exception as exc:  # noqa: BLE001 — surface clear startup error
        print(f"WARNING: failed to load Production model: {exc}")
        _model, _model_version = None, None
    yield


app = FastAPI(
    title="Telco Churn Prediction API",
    description="Serves the MLflow Production-registered churn classifier.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, Any]:
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train and promote a model first.",
        )
    return {
        "status": "ok",
        "model_name": REGISTERED_MODEL_NAME,
        "model_uri": MODEL_URI,
        "model_version": _model_version,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train and promote a model first.",
        )
    if not request.customers:
        raise HTTPException(status_code=400, detail="customers list is empty")

    frame = pd.DataFrame([c.model_dump() for c in request.customers])
    try:
        proba = _model.predict_proba(frame)[:, 1]
        preds = _model.predict(frame)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

    predictions = [
        Prediction(
            churn_probability=float(p),
            churn_prediction=int(y),
            churn_label="Yes" if int(y) == 1 else "No",
        )
        for p, y in zip(proba, preds)
    ]
    return PredictResponse(
        model_name=REGISTERED_MODEL_NAME,
        model_version=_model_version,
        predictions=predictions,
    )

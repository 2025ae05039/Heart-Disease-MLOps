"""Educational FastAPI inference service for the saved UCI heart-disease model."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.monitoring import log_prediction

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "artifacts" / "models" / "model.joblib"
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
METADATA_PATH = ROOT / "artifacts" / "models" / "metadata.json"

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
CATEGORICAL = {"sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"}

class HeartFeatures(BaseModel):
    """Input schema matching the feature order/names used by the notebook."""
    model_config = ConfigDict(extra="forbid")
    age: float = Field(ge=1, le=120)
    sex: str = Field(pattern=r"^(0|1)$")
    cp: str = Field(pattern=r"^[1-4]$")
    trestbps: float = Field(gt=0, le=300)
    chol: float = Field(gt=0, le=800)
    fbs: str = Field(pattern=r"^(0|1)$")
    restecg: str = Field(pattern=r"^[0-2]$")
    thalach: float = Field(gt=0, le=300)
    exang: str = Field(pattern=r"^(0|1)$")
    oldpeak: float = Field(ge=0, le=15)
    slope: str = Field(pattern=r"^[1-3]$")
    ca: str = Field(pattern=r"^[0-3]$")
    thal: str = Field(pattern=r"^(3|6|7)$")

    def as_model_frame(self) -> pd.DataFrame:
        payload: dict[str, Any] = self.model_dump()
        # The training pipeline expects coded categorical features as strings.
        for column in CATEGORICAL:
            payload[column] = str(payload[column])
        return pd.DataFrame([payload], columns=FEATURES)

app = FastAPI(
    title="Heart Disease MLOps Demo API",
    version="1.0.0",
    description=(
        "Educational classification demo only. Not for medical diagnosis or treatment decisions."
    ),
)

_model = None

def load_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model artifact not found: {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    return _model

@app.get("/health")
def health():
    """Health check; reports whether the artifact exists without running a prediction."""
    return {"status": "ok", "model_artifact_exists": MODEL_PATH.exists()}

@app.get("/metadata")
def metadata():
    if not METADATA_PATH.exists():
        raise HTTPException(status_code=404, detail="Model metadata not found")
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))

@app.post("/predict")
def predict(features: HeartFeatures):
    try:
        model = load_model()
        frame = features.as_model_frame()
        prediction = int(model.predict(frame)[0])
        probability = None
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(frame)[0][1])
        log_prediction(prediction=prediction, probability=probability, valid=True)
        return {
            "prediction": prediction,
            "label": "presence" if prediction == 1 else "absence",
            "probability_presence": probability,
            "educational_use_only": True,
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        log_prediction(prediction=None, probability=None, valid=False)
        raise HTTPException(status_code=500, detail="Prediction failed; inspect server logs.") from exc

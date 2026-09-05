"""
API de inferencia DiabetesRisk — Entrega 2/3.

Sirve el modelo empaquetado (mlflow/model.pkl) a través de un endpoint REST,
además de exponer información de modelamiento (métricas, matrices de
confusión e importancia de variables) para que el tablero pueda mostrar
detalle sin cargar el modelo directamente.

Ejecutar:
    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
import json
import joblib
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "mlflow" / "model.pkl"
METRICS_PATH = BASE_DIR / "mlflow" / "metrics.json"

app = FastAPI(
    title="DiabetesRisk API",
    description="Sirve predicciones de riesgo de diabetes/prediabetes e información de modelamiento.",
    version="1.1.0",
)

_bundle = None


def get_bundle():
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=503, detail=f"Modelo no encontrado en {MODEL_PATH}. Corra mlflow/train.py primero.")
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def get_metrics_file():
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=503, detail=f"Métricas no encontradas en {METRICS_PATH}. Corra mlflow/train.py primero.")
    with open(METRICS_PATH) as f:
        return json.load(f)


class RiskInput(BaseModel):
    HighBP: int = Field(..., ge=0, le=1, description="Presión arterial alta (0=No, 1=Sí)")
    HighChol: int = Field(..., ge=0, le=1, description="Colesterol alto (0=No, 1=Sí)")
    BMI: float = Field(..., ge=10, le=100, description="Índice de masa corporal")
    Smoker: int = Field(..., ge=0, le=1)
    Stroke: int = Field(..., ge=0, le=1)
    HeartDiseaseorAttack: int = Field(..., ge=0, le=1)
    PhysActivity: int = Field(..., ge=0, le=1)
    GenHlth: int = Field(..., ge=1, le=5, description="1=Excelente ... 5=Mala")
    DiffWalk: int = Field(..., ge=0, le=1)
    Age: int = Field(..., ge=1, le=13, description="Categoría de edad BRFSS (1=18-24 ... 13=80+)")
    Income: int = Field(..., ge=1, le=8)

    class Config:
        json_schema_extra = {
            "example": {
                "HighBP": 1, "HighChol": 1, "BMI": 32.0, "Smoker": 0, "Stroke": 0,
                "HeartDiseaseorAttack": 0, "PhysActivity": 1, "GenHlth": 4,
                "DiffWalk": 0, "Age": 9, "Income": 5,
            }
        }


class RiskOutput(BaseModel):
    risk_probability: float
    risk_level: str
    model_name: str


@app.get("/health")
def health():
    bundle = get_bundle()
    return {"status": "ok", "model_name": bundle.get("model_name", "unknown")}


@app.post("/predict", response_model=RiskOutput)
def predict(payload: RiskInput):
    bundle = get_bundle()
    model, features = bundle["model"], bundle["features"]

    row = pd.DataFrame([payload.dict()])[features]
    proba = float(model.predict_proba(row)[0, 1])

    if proba < 0.20:
        level = "bajo"
    elif proba < 0.45:
        level = "moderado"
    else:
        level = "alto"

    return RiskOutput(
        risk_probability=proba,
        risk_level=level,
        model_name=bundle.get("model_name", "unknown"),
    )


@app.get("/metrics")
def metrics():
    """Métricas y matriz de confusión de las 4 corridas, más el modelo seleccionado."""
    return get_metrics_file()


@app.get("/model-info")
def model_info():
    """Hiperparámetros e importancia de variables del modelo activo."""
    bundle = get_bundle()
    model, features = bundle["model"], bundle["features"]

    info = {
        "model_name": bundle.get("model_name", "unknown"),
        "features": features,
        "hyperparameters": {k: (v if isinstance(v, (int, float, str, bool)) or v is None else str(v))
                             for k, v in model.get_params().items()},
    }
    if hasattr(model, "feature_importances_"):
        info["feature_importances"] = dict(zip(features, model.feature_importances_.tolist()))
    else:
        info["feature_importances"] = None

    return info
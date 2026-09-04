"""
API de inferencia DiabetesRisk — Entrega 2/3.

Sirve el modelo empaquetado (src/model.pkl) a través de un endpoint REST.
El tablero (dashboard/app.py) consume esta API en vez de cargar el modelo
directamente.

Ejecutar:
    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
import joblib
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parent.parent / "src" / "model.pkl"

app = FastAPI(
    title="DiabetesRisk API",
    description="Sirve predicciones de riesgo de diabetes/prediabetes a partir de variables demográficas y de estilo de vida.",
    version="1.0.0",
)

_bundle = None


def get_bundle():
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=503, detail=f"Modelo no encontrado en {MODEL_PATH}. Corra src/train.py primero.")
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


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

"""
API de inferencia para DiabetesRisk (Semana 6+).
Placeholder: reemplazar la carga del modelo cuando esté empaquetado.
"""
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="DiabetesRisk API")


class RiskInput(BaseModel):
    age: int
    bmi: float
    high_bp: bool
    high_chol: bool
    gen_health: int  # 1 (excelente) a 5 (mala)
    phys_activity: bool
    diff_walk: bool


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: RiskInput):
    # TODO Semana 6: cargar modelo empaquetado (joblib/mlflow) y predecir de verdad
    return {"risk_probability": 0.0, "note": "modelo pendiente de empaquetar"}

# DiabetesRisk — Micro-proyecto MAIA

Tamizaje temprano de riesgo de diabetes tipo 2 a partir de variables
demográficas y de estilo de vida, sin exámenes de laboratorio.

Equipo: Jared Foster Orduz · Jeferson David Vargas Toca ·
Andres Felipe Florez Garces · Oscar Enrique Morillo

## Estructura del repositorio

```
data/          datasets versionados con DVC (no se versiona el .csv en Git)
notebooks/     exploración de datos (EDA)
src/           procesamiento, entrenamiento y empaquetado del modelo
api/           servicio de inferencia (FastAPI)
dashboard/     tablero (Streamlit)
docker/        Dockerfiles y docker-compose.yml
```

## Dataset

Diabetes Health Indicators Dataset (BRFSS 2015, CDC), 253,680 registros,
22 variables. Fuente: Kaggle / UCI ML Repository.

## Puesta en marcha (local)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Versionamiento de datos (DVC + AWS S3)

```bash
dvc init
dvc remote add -d aws-remote s3://<bucket-diabetesrisk-dvcstore>
dvc add data/diabetes_binary_health_indicators_BRFSS2015.csv
git add data/*.dvc .dvc/config .gitignore
git commit -m "Versionar dataset inicial con DVC"
dvc push
```

## Estado del proyecto

- [x] Semana 1-2: problema, pregunta de negocio, maqueta.
- [x] Semana 3: EDA inicial, repos Git/DVC.
- [ ] Semana 4+: modelos, MLflow, API, tablero, despliegue Docker/AWS.

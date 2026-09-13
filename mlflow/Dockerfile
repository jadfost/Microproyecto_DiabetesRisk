FROM python:3.11-slim

# Instala las dependencias de MLflow / entrenamiento
WORKDIR /app/mlflow
COPY mlflow/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY mlflow/train.py .

# El WORKDIR final es /app para que las rutas relativas de train.py
# ("data/...", "mlflow/model.pkl") apunten a /app/data y /app/mlflow.
WORKDIR /app

EXPOSE 5000

# Comando por defecto: levantar el tracking server de MLflow.
# El servicio "trainer" en docker-compose sobreescribe este CMD
# para correr `python mlflow/train.py` en su lugar.
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "sqlite:////app/mlflow-data/mlflow.db", \
     "--default-artifact-root", "/app/mlflow-data/artifacts"]

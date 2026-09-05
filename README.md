# DiabetesRisk — Micro-proyecto MAIA

Tamizaje temprano de riesgo de diabetes tipo 2 a partir de variables
demográficas y de estilo de vida, sin exámenes de laboratorio.

**Equipo:** Jared Foster Orduz · Jeferson David Vargas Toca · Andres Felipe Florez Garces · Oscar Enrique Morillo

**Enlaces (instancia EC2, IP pública `44.204.142.207`):**
- Tablero (Streamlit): http://44.204.142.207:8501
- API (docs interactivas): http://44.204.142.207:8000/docs
- MLflow: http://44.204.142.207:5000
- Repositorio: https://github.com/jadfost/Microproyecto_DiabetesRisk

---

## 1. Estructura del repositorio

```
Microproyecto_DiabetesRisk/
├── data/           # dataset versionado con DVC (no se sube a Git directamente)
├── mlflow/
│   ├── train.py    # entrenamiento de los 4 modelos + registro en MLflow
│   ├── requirements.txt # dependencias del entorno de entrenamiento
│   ├── model.pkl   # modelo ganador empaquetado (se genera al correr train.py)
│   └── metrics.json # métricas + matriz de confusión de las 4 corridas
├── api/
│   ├── main.py     # API FastAPI que sirve el modelo
│   └── requirements.txt
├── dashboard/
│   ├── app.py      # tablero Streamlit (Inicio + Detalle del modelamiento)
│   └── requirements.txt
├── docker/         # Dockerfiles y docker-compose.yml (semana 6+)
└── dvc.yaml, .dvc/ # configuración de DVC
```

**Regla de oro de la arquitectura:** el tablero **nunca** carga el modelo directamente — siempre habla con la API por HTTP (`API_URL`, por defecto `http://localhost:8000`). La API es la única que abre `mlflow/model.pkl`.

---

## 2. Los 3 entornos virtuales (por qué existen)

MLflow, la API y el tablero necesitan versiones de librerías que **chocan entre sí** (el caso más molesto: Streamlit necesita `protobuf>=5.26`, MLflow necesita `protobuf<5`). Para no pelear con eso, cada servicio vive en su propio entorno virtual de Python. Es exactamente el mismo problema que resuelven los contenedores Docker — esto es un ensayo de esa separación mientras llegamos a esa parte del proyecto.

| Entorno | Para qué | Carpeta sugerida |
|---|---|---|
| `venv-train` | Entrenar modelos y correr el servidor MLflow | `~/venv-train` |
| `venv-api` | Correr la API (FastAPI) | `~/venv-api` |
| `venv-dashboard` | Correr el tablero (Streamlit) | `~/venv-dashboard` |

### Crear los 3 (solo la primera vez)

```bash
cd ~/Microproyecto_DiabetesRisk

# Si no lo tienen instalado en la instancia:
sudo apt install -y python3.12-venv

python3 -m venv ~/venv-train
python3 -m venv ~/venv-api
python3 -m venv ~/venv-dashboard
```

### Instalar dependencias en cada uno (solo la primera vez, o si cambian los requirements)

```bash
source ~/venv-train/bin/activate
pip install -r mlflow/requirements.txt
deactivate

source ~/venv-api/bin/activate
pip install -r api/requirements.txt
deactivate

source ~/venv-dashboard/bin/activate
pip install -r dashboard/requirements.txt
deactivate
```

---

## 3. Cómo correr cada cosa

Cada componente necesita **su propia terminal** (o su propia sesión de EC2 Instance Connect / pestaña de tmux). Si solo tienen una terminal, usen `nohup ... &` como se muestra abajo para dejarlo corriendo en segundo plano y liberar la terminal.

### 3.1 Servidor MLflow (primero que todo)

```bash
source ~/venv-train/bin/activate
cd ~/Microproyecto_DiabetesRisk
nohup mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --workers 1 > mlflow.log 2>&1 &
sleep 5 && cat mlflow.log   # confirmar que dice "Listening at: http://0.0.0.0:5000"
```

Ver en el navegador: http://44.204.142.207:5000

### 3.2 Entrenar los modelos (genera model.pkl y metrics.json)

Con el servidor MLflow ya corriendo:

```bash
source ~/venv-train/bin/activate
cd ~/Microproyecto_DiabetesRisk
export MLFLOW_TRACKING_URI=http://localhost:5000
python3 mlflow/train.py
```

Esto entrena 4 modelos, registra cada uno en MLflow, y guarda el modelo ganador en `mlflow/model.pkl` junto con `mlflow/metrics.json`. **Hay que correr esto antes de levantar la API**, porque la API necesita `model.pkl` para arrancar bien.

### 3.3 API (FastAPI)

```bash
source ~/venv-api/bin/activate
cd ~/Microproyecto_DiabetesRisk
nohup uvicorn api.main:app --host 0.0.0.0 --port 8000 > api.log 2>&1 &
sleep 5 && cat api.log   # confirmar "Application startup complete"
curl http://localhost:8000/health   # debe responder {"status":"ok",...}
```

Documentación interactiva (para probar `/predict` a mano): http://44.204.142.207:8000/docs

### 3.4 Tablero (Streamlit)

Con la API ya corriendo:

```bash
source ~/venv-dashboard/bin/activate
cd ~/Microproyecto_DiabetesRisk
export API_URL=http://localhost:8000
nohup streamlit run dashboard/app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true > streamlit.log 2>&1 &
sleep 8 && cat streamlit.log
```

Ver en el navegador: http://44.204.142.207:8501

---

## 4. Orden de arranque resumido

```
1. MLflow          (3.1)
2. Entrenar modelo (3.2)  ← necesita MLflow corriendo
3. API             (3.3)  ← necesita model.pkl (paso 2)
4. Tablero         (3.4)  ← necesita la API corriendo
```

Si alguno de estos ya está corriendo (por ejemplo, MLflow lo dejó corriendo otro compañero), no hace falta repetirlo — solo confirmen con `curl http://localhost:<puerto>/health` (API) o revisando la URL en el navegador (MLflow y tablero).

---

## 5. Comandos útiles para diagnosticar

```bash
# Ver qué está corriendo
ps aux | grep -E "mlflow|uvicorn|streamlit"

# Ver logs de cada servicio
cat ~/Microproyecto_DiabetesRisk/mlflow.log
cat ~/Microproyecto_DiabetesRisk/api.log
cat ~/Microproyecto_DiabetesRisk/streamlit.log

# Matar un proceso colgado (buscar el PID con ps aux de arriba)
kill <PID>

# Revalidar sesión AWS si algo con S3/DVC falla por token expirado
aws sso login --profile Universidad-Developer-373665157741
```

---

## 6. Sincronizar cambios (Git + DVC)

```bash
# Traer cambios de los demás
git pull

# Traer el dataset si no está (o si cambió)
python3 -m dvc pull   # o "dvc pull" si ya está en el PATH

# Subir cambios propios
git add <archivos>
git commit -m "mensaje descriptivo"
git push origin main
```

---
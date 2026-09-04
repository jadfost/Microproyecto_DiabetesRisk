"""
Tablero DiabetesRisk — Entrega 2/3.

Formulario de evaluación de riesgo individual (consume la API REST, que a su
vez sirve el modelo empaquetado) + panel de estadísticas descriptivas de la
población, siguiendo la maqueta diseñada en la Entrega 1.

Variables de entorno:
    API_URL: URL base de la API (por defecto http://localhost:8000)

Ejecutar:
    streamlit run dashboard/app.py --server.address 0.0.0.0 --server.port 8501
"""
import os
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="DiabetesRisk", page_icon="🩺", layout="wide")

API_URL = os.getenv("API_URL", "http://localhost:8000")
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "diabetes_binary_health_indicators_BRFSS2015.csv"

AGE_LABELS = {
    1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44", 6: "45-49",
    7: "50-54", 8: "55-59", 9: "60-64", 10: "65-69", 11: "70-74", 12: "75-79", 13: "80+",
}
GENHLTH_LABELS = {1: "Excelente", 2: "Muy buena", 3: "Buena", 4: "Regular", 5: "Mala"}


@st.cache_data(ttl=30)
def check_api_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@st.cache_data
def load_population_stats():
    df = pd.read_csv(DATA_PATH)
    return df


st.markdown(
    "<h2 style='margin-bottom:0;'>🩺 DiabetesRisk</h2>"
    "<p style='color:#8C8A85;margin-top:0;'>Tamizaje temprano de riesgo de diabetes tipo 2</p>",
    unsafe_allow_html=True,
)

health = check_api_health()
if health.get("status") != "ok":
    st.error(f"No se pudo conectar con la API en {API_URL}. Detalle: {health.get('detail')}")
    st.stop()

model_name = health.get("model_name", "modelo")

col_form, col_stats = st.columns([1, 1.4], gap="large")

with col_form:
    st.subheader("Evaluación de riesgo")
    st.caption(f"Conectado a la API: {API_URL}  ·  Modelo activo: {model_name.replace('_', ' ').title()}")
    with st.form("risk_form"):
        age_group = st.selectbox(
            "Edad", options=list(AGE_LABELS.keys()), format_func=lambda k: AGE_LABELS[k], index=6
        )
        bmi = st.number_input("Índice de masa corporal (IMC)", 12.0, 60.0, 27.5, step=0.5)
        high_bp = st.selectbox("Presión arterial alta", ["No", "Sí"]) == "Sí"
        high_chol = st.selectbox("Colesterol alto", ["No", "Sí"]) == "Sí"
        gen_health = st.select_slider(
            "Salud general percibida", options=list(GENHLTH_LABELS.keys()),
            value=2, format_func=lambda k: GENHLTH_LABELS[k]
        )
        phys_activity = st.selectbox("Actividad física en el último mes", ["Sí", "No"]) == "Sí"
        diff_walk = st.selectbox("Dificultad seria para caminar o subir escaleras", ["No", "Sí"]) == "Sí"
        smoker = st.selectbox("¿Ha fumado al menos 100 cigarrillos en su vida?", ["No", "Sí"]) == "Sí"
        stroke = st.selectbox("¿Ha tenido un accidente cerebrovascular?", ["No", "Sí"]) == "Sí"
        heart_disease = st.selectbox("¿Enfermedad coronaria o infarto previo?", ["No", "Sí"]) == "Sí"
        income = st.slider("Nivel de ingresos (1=más bajo, 8=más alto)", 1, 8, 5)

        submitted = st.form_submit_button("Calcular riesgo", use_container_width=True)

    if submitted:
        payload = {
            "HighBP": int(high_bp), "HighChol": int(high_chol), "BMI": bmi,
            "Smoker": int(smoker), "Stroke": int(stroke),
            "HeartDiseaseorAttack": int(heart_disease), "PhysActivity": int(phys_activity),
            "GenHlth": gen_health, "DiffWalk": int(diff_walk), "Age": age_group, "Income": income,
        }
        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
            resp.raise_for_status()
            result = resp.json()
            proba = result["risk_probability"]
            level = result["risk_level"]
            pct = proba * 100

            color = {"bajo": "#1D9E75", "moderado": "#D8A400", "alto": "#D85A30"}[level]
            st.markdown(
                f"<div style='background:{color}22;border-radius:10px;padding:14px 18px;margin-top:10px;'>"
                f"<span style='color:{color};font-size:13px;'>Resultado (vía API)</span><br>"
                f"<span style='color:{color};font-size:26px;font-weight:700;'>Riesgo {level.capitalize()} · {pct:.1f}%</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.caption(
                "Este resultado es una estimación de un prototipo académico, no un diagnóstico médico. "
                "Ante un riesgo moderado o alto, se recomienda consultar a un profesional de la salud."
            )
        except Exception as e:
            st.error(f"Error al consultar la API: {e}")

with col_stats:
    st.subheader("Estadísticas de la población (BRFSS 2015)")
    try:
        df = load_population_stats()
        c1, c2, c3 = st.columns(3)
        c1.metric("Registros analizados", f"{len(df):,}")
        c2.metric("% con diabetes/prediab.", f"{df['Diabetes_binary'].mean()*100:.1f}%")
        c3.metric("Modelo activo", model_name.replace("_", " ").title())

        st.markdown("**Prevalencia de diabetes por rango de IMC**")
        bmi_bin = pd.cut(df["BMI"], bins=[0, 18.5, 25, 30, 35, 40, 100],
                          labels=["<18.5", "18.5-25", "25-30", "30-35", "35-40", "40+"])
        rate = df.groupby(bmi_bin, observed=True)["Diabetes_binary"].mean() * 100
        fig, ax = plt.subplots(figsize=(6, 2.6))
        ax.bar(rate.index.astype(str), rate.values, color="#0C447C")
        ax.set_ylabel("% con diabetes/prediab.")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig, use_container_width=True)

        st.markdown("**Prevalencia de diabetes por grupo de edad**")
        df["AgeGroup"] = df["Age"].map(AGE_LABELS)
        rate_age = df.groupby("AgeGroup", observed=True)["Diabetes_binary"].mean().reindex(AGE_LABELS.values()) * 100
        fig2, ax2 = plt.subplots(figsize=(6, 2.6))
        ax2.plot(rate_age.index, rate_age.values, color="#534AB7", marker="o")
        ax2.set_ylabel("% con diabetes/prediab.")
        ax2.spines[["top", "right"]].set_visible(False)
        plt.xticks(rotation=40, ha="right")
        st.pyplot(fig2, use_container_width=True)
    except FileNotFoundError:
        st.info("Dataset no encontrado en esta máquina (data/*.csv). Corra `dvc pull` para traerlo.")

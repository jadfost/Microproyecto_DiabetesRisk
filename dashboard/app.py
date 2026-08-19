"""
Tablero DiabetesRisk (Semana 6+).
Placeholder: conectar con la API real cuando esté desplegada.
"""
import os
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("DiabetesRisk — Evaluación de riesgo de diabetes")

with st.form("risk_form"):
    age = st.number_input("Edad", 18, 100, 45)
    bmi = st.number_input("IMC", 12.0, 60.0, 27.5)
    high_bp = st.selectbox("Presión arterial alta", ["No", "Sí"]) == "Sí"
    high_chol = st.selectbox("Colesterol alto", ["No", "Sí"]) == "Sí"
    gen_health = st.slider("Salud general (1=excelente, 5=mala)", 1, 5, 2)
    phys_activity = st.selectbox("Actividad física semanal", ["Sí", "No"]) == "Sí"
    diff_walk = st.selectbox("Dificultad para caminar", ["No", "Sí"]) == "Sí"
    submitted = st.form_submit_button("Calcular riesgo")

if submitted:
    st.info("Conexión con la API pendiente (Semana 6). Placeholder de UI listo.")

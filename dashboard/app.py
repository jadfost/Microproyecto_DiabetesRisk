"""
Tablero DiabetesRisk — Entrega 2/3.

Dos páginas:
- Inicio: formulario de evaluación individual (vía API) + gráficos de
  población que se actualizan dinámicamente según los parámetros ingresados.
- Detalle del modelamiento: explicación accesible de cómo se entrenó y
  evaluó el modelo (comparación de modelos, matriz de confusión, importancia
  de variables), consumiendo la API.

Variables de entorno:
    API_URL: URL base de la API (por defecto http://localhost:8000)

Ejecutar:
    streamlit run dashboard/app.py --server.address 0.0.0.0 --server.port 8501
"""
import os
import pandas as pd
import numpy as np
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
BMI_BINS = [0, 18.5, 25, 30, 35, 40, 100]
BMI_LABELS = ["<18.5", "18.5-25", "25-30", "30-35", "35-40", "40+"]

NAVY = "#0C447C"
CORAL = "#D85A30"
TEAL = "#1D9E75"
PURPLE = "#534AB7"
GRAY = "#8C8A85"


# ---------- Llamadas a la API ----------

@st.cache_data(ttl=30)
def check_api_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@st.cache_data(ttl=60)
def fetch_metrics():
    r = requests.get(f"{API_URL}/metrics", timeout=5)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=60)
def fetch_model_info():
    r = requests.get(f"{API_URL}/model-info", timeout=5)
    r.raise_for_status()
    return r.json()


@st.cache_data
def load_population_stats():
    return pd.read_csv(DATA_PATH)


def style_ax(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors="#4a4a4a", labelsize=9)
    ax.grid(axis="y", color="#E4E2DC", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


MODEL_LABELS = {
    "logreg_baseline": "Regresión Logística (base)",
    "logreg_smote": "Regresión Logística + SMOTE",
    "random_forest_balanced": "Random Forest (class_weight)",
    "random_forest_smote": "Random Forest + SMOTE",
}


# ---------- Layout general ----------

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

page = st.sidebar.radio("Menú", ["Inicio", "Detalle del modelamiento"])
st.sidebar.markdown("---")
st.sidebar.caption(
    f"**API interna:** `http://44.204.142.207:8000/` (misma instancia, no requiere salir a internet)\n\n"
    f"**Modelo activo:** {model_name.replace('_', ' ').title()}\n\n"
    f"La API también es accesible públicamente para pruebas (ver `http://44.204.142.207:8000/docs`)."
)


# ============================================================
# PÁGINA 1: INICIO
# ============================================================
if page == "Inicio":
    # Slot reservado arriba de todo para el resultado de "Calcular riesgo".
    # Aunque el botón vive más abajo (dentro del formulario), lo que se
    # escriba aquí dentro aparecerá en esta posición: así el resultado
    # queda bien visible sin tener que hacer scroll hasta el final del form.
    result_slot = st.container()

    try:
        _population_df_preview = load_population_stats()
        population_rate = _population_df_preview["Diabetes_binary"].mean() * 100
    except FileNotFoundError:
        population_rate = None

    col_form, col_stats = st.columns([1, 1.4], gap="large")

    with col_form:
        st.subheader("Evaluación de riesgo")
        st.caption("Los gráficos de la derecha se actualizan mientras ajustas los valores.")

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

        calc = st.button("Calcular riesgo", use_container_width=True, type="primary")

        if calc:
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
                proba, level = result["risk_probability"], result["risk_level"]
                pct = proba * 100
                color = {"bajo": TEAL, "moderado": "#D8A400", "alto": CORAL}[level]
                icon = {"bajo": "✅", "moderado": "⚠️", "alto": "🚨"}[level]
                level_msg = {
                    "bajo": "Tu perfil actual cae en el rango de riesgo más bajo estimado por el modelo.",
                    "moderado": "Tu perfil muestra un riesgo intermedio: vale la pena prestarle atención.",
                    "alto": "Tu perfil coincide con el patrón de mayor riesgo detectado por el modelo.",
                }[level]

                diff_txt = ""
                if population_rate is not None:
                    diff = pct - population_rate
                    comp = "por encima" if diff >= 0 else "por debajo"
                    diff_txt = (
                        f"Esto es <b>{abs(diff):.1f} puntos {comp}</b> del promedio de la población "
                        f"analizada ({population_rate:.1f}%)."
                    )

                with result_slot:
                    st.markdown(
                        f"""
                        <div style='background:{color}18;border:1.5px solid {color};
                                    border-radius:14px;padding:20px 26px;margin-bottom:4px;'>
                          <div style='display:flex;align-items:center;gap:12px;'>
                            <span style='font-size:32px;line-height:1;'>{icon}</span>
                            <div>
                              <div style='color:{color};font-size:13px;font-weight:700;letter-spacing:.4px;'>
                                RESULTADO DE LA EVALUACIÓN
                              </div>
                              <div style='color:{color};font-size:32px;font-weight:800;line-height:1.15;'>
                                Riesgo {level.capitalize()} · {pct:.1f}%
                              </div>
                            </div>
                          </div>
                          <div style='margin-top:14px;background:#00000030;border-radius:6px;
                                      height:10px;overflow:hidden;'>
                            <div style='width:{min(pct, 100):.1f}%;background:{color};height:100%;'></div>
                          </div>
                          <p style='margin:14px 0 0 0;color:#E4E2DC;font-size:14.5px;line-height:1.5;'>
                            {level_msg} {diff_txt}
                          </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        "Esta es una estimación de un prototipo académico, no un diagnóstico médico. "
                        "Ante un riesgo moderado o alto, consulte a un profesional de la salud."
                    )
            except Exception as e:
                with result_slot:
                    st.error(f"Error al consultar la API: {e}")

    with col_stats:
        st.subheader("Tu perfil frente a la población (BRFSS 2015)")
        try:
            df = load_population_stats()
            c1, c2, c3 = st.columns(3)
            c1.metric("Registros analizados", f"{len(df):,}")
            c2.metric("% con diabetes/prediab.", f"{df['Diabetes_binary'].mean()*100:.1f}%")
            c3.metric("Modelo activo", model_name.replace("_", " ").title())

            # --- Gráfico dinámico 1: IMC, resalta el bin del usuario ---
            st.markdown(f"**Prevalencia de diabetes por rango de IMC** — tu IMC: {bmi:.1f}")
            bmi_bin_series = pd.cut(df["BMI"], bins=BMI_BINS, labels=BMI_LABELS)
            rate = df.groupby(bmi_bin_series, observed=True)["Diabetes_binary"].mean() * 100
            user_bin = pd.cut([bmi], bins=BMI_BINS, labels=BMI_LABELS)[0]

            fig, ax = plt.subplots(figsize=(6, 2.8))
            colors = [CORAL if lbl == str(user_bin) else NAVY for lbl in rate.index.astype(str)]
            bars = ax.bar(rate.index.astype(str), rate.values, color=colors, zorder=3)
            for lbl, bar in zip(rate.index.astype(str), bars):
                if lbl == str(user_bin):
                    ax.annotate("Tú estás aquí", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                                xytext=(0, 12), textcoords="offset points", ha="center",
                                fontsize=9, color=CORAL, fontweight="bold")
            ax.set_ylabel("% con diabetes/prediab.")
            style_ax(ax)
            st.pyplot(fig, use_container_width=True)
            st.caption(
                f"Este gráfico agrupa a **toda la población por rango de IMC** (no por edad). "
                f"La barra naranja **'Tú estás aquí'** marca el rango donde cae tu IMC, que ingresaste "
                f"como **{bmi:.1f}** — por eso puede resaltar un rango distinto al de tu edad, son dos "
                f"variables independientes."
            )

            # --- Gráfico dinámico 2: edad, resalta el grupo del usuario ---
            st.markdown(f"**Prevalencia de diabetes por grupo de edad** — tu grupo: {AGE_LABELS[age_group]}")
            df["AgeGroup"] = df["Age"].map(AGE_LABELS)
            rate_age = df.groupby("AgeGroup", observed=True)["Diabetes_binary"].mean().reindex(AGE_LABELS.values()) * 100
            user_age_label = AGE_LABELS[age_group]

            fig2, ax2 = plt.subplots(figsize=(6, 2.8))
            ax2.plot(rate_age.index, rate_age.values, color=PURPLE, linewidth=2, zorder=3)
            sizes = [90 if lbl == user_age_label else 25 for lbl in rate_age.index]
            colors2 = [CORAL if lbl == user_age_label else PURPLE for lbl in rate_age.index]
            ax2.scatter(rate_age.index, rate_age.values, s=sizes, color=colors2, zorder=4)
            if user_age_label in rate_age.index:
                yv = rate_age[user_age_label]
                ax2.annotate("Tú", xy=(user_age_label, yv), xytext=(0, 10),
                             textcoords="offset points", ha="center", fontsize=9, color=CORAL, fontweight="bold")
            ax2.set_ylabel("% con diabetes/prediab.")
            style_ax(ax2)
            plt.xticks(rotation=40, ha="right")
            st.pyplot(fig2, use_container_width=True)
            st.caption(
                f"Este otro gráfico agrupa a la población por **grupo de edad** (no por IMC). "
                f"El punto naranja **'Tú'** marca tu grupo de edad, **{AGE_LABELS[age_group]}**, "
                f"seleccionado en el formulario — independiente del gráfico de IMC de arriba."
            )

        except FileNotFoundError:
            st.info("Dataset no encontrado en esta máquina (data/*.csv). Corra `dvc pull` para traerlo.")


# ============================================================
# PÁGINA 2: DETALLE DEL MODELAMIENTO
# ============================================================
else:
    st.subheader("Detalle del modelamiento")
    st.caption(
        "Esta página explica, de forma sencilla, cómo se construyó y evaluó el modelo que usa el tablero. "
        "Toda la información viene directamente de la API (endpoints /metrics y /model-info)."
    )

    try:
        metrics_data = fetch_metrics()
        model_info = fetch_model_info()
    except Exception as e:
        st.error(f"No se pudo obtener el detalle del modelamiento desde la API: {e}")
        st.stop()

    results = metrics_data["results"]
    selected = metrics_data["selected_model"]

    with st.expander("¿Cómo se entrenó el modelo?", expanded=True):
        st.markdown(
            "- Se probaron **4 modelos** distintos y se compararon entre sí.\n"
            "- Solo el **13.9%** de las personas en los datos tienen diabetes o prediabetes — un dataset "
            "desbalanceado. Si el modelo simplemente dijera \"nadie tiene diabetes\" acertaría el 86% de las veces, "
            "¡pero no serviría de nada!\n"
            "- Por eso se probaron dos técnicas para tratar de balancear los datos o al menos \"enseñarle\" al modelo a prestar más atención a los casos "
            "positivos: **SMOTE** que básicamente es crear ejemplos sintéticos de la clase minoritaria y **class_weight** que"
            " es penalizar más los errores sobre esa clase.\n"
            f"- El modelo ganador fue **{MODEL_LABELS.get(selected, selected)}**, elegido por tener el mejor "
            "*recall* (mayor capacidad de detectar los casos reales de riesgo).  Es importante en este caso tener el mejor *recall* posible"
            " aunque eso implique que el modelo genere más falsos positivos (personas que no tienen riesgo real pero que el modelo predice como de riesgo).\n\n"
            "A continuación se muestran los resultados de la comparación de los 4 modelos, la matriz de confusión del modelo ganador y la importancia de variables."
        )

    st.markdown("### Comparación de los 4 modelos")
    comp_df = pd.DataFrame(results).T[["accuracy", "precision", "recall", "f1", "roc_auc"]]
    comp_df.index = [MODEL_LABELS.get(i, i) for i in comp_df.index]
    st.dataframe(comp_df.style.format("{:.3f}").highlight_max(axis=0, color="#1D9E7533"), use_container_width=True)

    fig, ax = plt.subplots(figsize=(12, 3))
    x = np.arange(len(comp_df))
    width = 0.2
    metric_cols = ["accuracy", "recall", "f1", "roc_auc"]
    colors_m = [NAVY, CORAL, TEAL, PURPLE]
    for i, (m, c) in enumerate(zip(metric_cols, colors_m)):
        ax.bar(x + i * width - 1.5 * width, comp_df[m].values, width, label=m, color=c, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(comp_df.index, fontsize=1)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=4, frameon=False, fontsize=9)
    style_ax(ax)
    st.pyplot(fig, use_container_width=True)

    st.markdown("Como se puede observar el modelo que mejor recall tiene es el Random Forest (class_weight).")

    st.markdown(f"### Matriz de confusión — {MODEL_LABELS.get(selected, selected)}")
    with st.expander("¿Qué es una matriz de confusión?"):
        st.markdown(
            "Compara lo que el modelo predijo contra la realidad. Nos interesa especialmente los "
            "**falsos negativos** (personas con riesgo real que el modelo no detectó), porque en un tamizaje "
            "de salud es más grave dejar pasar un caso real que remitir de más a alguien a un examen."
        )
    #cm = results[selected]["confusion_matrix"]
    #cm_df = pd.DataFrame(
    #    cm,
    #    index=["Real: Sin diabetes", "Real: Con diabetes/prediab."],
    #    columns=["Predicho: Sin diabetes", "Predicho: Con diabetes/prediab."],
    #)
    #st.dataframe(cm_df, use_container_width=True)

    cm = np.array(results[selected]["confusion_matrix"])

    labels = ["Sin diabetes", "Con diabetes/prediab."]

    fig, ax = plt.subplots(figsize=(15, 3.5))
    # Pintar matriz
    im = ax.imshow(cm, cmap="Blues")
    # Ejes
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Valor real")
    ax.set_title("Matriz de confusión", fontweight="bold", pad=12)
    # Mostrar valores dentro de cada celda
    threshold = cm.max() / 2

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=7, fontweight="bold", color="white" if cm[i, j] > threshold else "black",)

    # Colorbar
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # Evitar cortes
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### Importancia de variables")
        fi = model_info.get("feature_importances")
        if fi:
            fi_series = pd.Series(fi).sort_values()
            fig3, ax3 = plt.subplots(figsize=(5, 4.2))
            ax3.barh(fi_series.index, fi_series.values, color=NAVY, zorder=3)
            style_ax(ax3)
            ax3.set_xlabel("Importancia (Gini)")
            st.pyplot(fig3, use_container_width=True)
            st.caption(
                f"Para el modelo, **{fi_series.idxmax()}** es la variable que más pesa al momento de "
                "estimar el riesgo, seguida por las siguientes en la lista."
            )
        else:
            st.info("Este modelo no expone importancia de variables.")

    with col_b:
        st.markdown("### Hiperparámetros del modelo")
        hp = model_info.get("hyperparameters", {})
        relevant = {k: v for k, v in hp.items() if k in (
            "n_estimators", "max_depth", "criterion", "max_features",
            "min_samples_split", "min_samples_leaf", "bootstrap", "class_weight",
        )}
        st.table(pd.DataFrame(relevant.items(), columns=["Parámetro", "Valor"]))
        st.caption("Valores obtenidos en vivo desde la API (endpoint /model-info), no están escritos a mano.")

    st.markdown("### Exploración de los datos")
    try:
        df = load_population_stats()
        corr = df.corr(numeric_only=True)["Diabetes_binary"].drop("Diabetes_binary").sort_values()
        selected_features = model_info.get("features", [])
        fig4, ax4 = plt.subplots(figsize=(8, 5.5))
        colors4 = [NAVY if (n in selected_features and v >= 0) else
                   (CORAL if (n in selected_features and v < 0) else "#C9C7C2")
                   for n, v in corr.items()]
        ax4.barh(corr.index, corr.values, color=colors4, zorder=3)
        style_ax(ax4)
        ax4.set_xlabel("Correlación de Pearson con Diabetes_binary")
        st.pyplot(fig4, use_container_width=True)
        st.caption(
            "En azul/naranja, las variables que finalmente se usaron en el modelo (mayor correlación con el "
            "riesgo real de diabetes); en gris, las que se descartaron por aportar poca información adicional."
        )
    except FileNotFoundError:
        st.info("Dataset no encontrado en esta máquina para calcular la exploración de datos.")
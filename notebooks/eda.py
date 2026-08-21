"""
Exploración de datos DiabetesRisk (Entrega 1).
Reproduce las figuras del reporte a partir de data/diabetes_binary_health_indicators_BRFSS2015.csv
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = Path("data/diabetes_binary_health_indicators_BRFSS2015.csv")
OUT_DIR = Path("notebooks/figs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(DATA_PATH)
    print(df.shape, "filas x columnas")
    print(df.isna().sum().sum(), "valores nulos")

    # ##oscar##
    # ------------------------------------
    # LIMPIEZA Y ESTADISTICAS DESCRIPTIVAS
    # ------------------------------------

    print(df.duplicated().sum(), "filas duplicadas")

    print("\nEstadisticas descriptivas:")
    print(df.describe().T)

    print("\nCantidad de registros por clase:")
    print(
        df["Diabetes_binary"]
        .value_counts()
        .sort_index()
    )

    print("\nPorcentaje por clase:")
    print(
        (
            df["Diabetes_binary"]
            .value_counts(normalize=True)
            .sort_index()
            * 100
        ).round(2)
    )
    # ##fin##

    print(df["Diabetes_binary"].value_counts(normalize=True))

    counts = df["Diabetes_binary"].value_counts().sort_index()
    fig, ax = plt.subplots()
    ax.bar(["Sin diabetes", "Prediabetes/diabetes"], counts.values)
    ax.set_title("Balance de clases")

    # ##oscar##
    # -----------------
    # BALANCE DE CLASES
    # -----------------

    percentages = counts / counts.sum() * 100

    for i, (count, percentage) in enumerate(
        zip(counts.values, percentages.values)
    ):
        ax.text(
            i,
            count,
            f"{percentage:.1f}%\n{count:,} registros",
            ha="center",
            va="bottom"
        )

    ax.set_ylabel("Numero de registros")
    # ##fin##

    fig.savefig(OUT_DIR / "balance_clases.png", dpi=150, bbox_inches="tight")

    # ##oscar##
    # ----------------------------
    # PREVALENCIA POR RANGO DE IMC
    # ----------------------------

    bmi_bins = [
        0,
        18.5,
        25,
        30,
        35,
        40,
        float("inf")
    ]

    bmi_labels = [
        "<18.5",
        "18.5-25",
        "25-30",
        "30-35",
        "35-40",
        "40+"
    ]

    df["BMI_range"] = pd.cut(
        df["BMI"],
        bins=bmi_bins,
        labels=bmi_labels,
        right=False
    )

    bmi_prevalencia = (
        df.groupby(
            "BMI_range",
            observed=False
        )["Diabetes_binary"]
        .mean()
        * 100
    )

    print("\nPrevalencia por rango de IMC (%):")
    print(bmi_prevalencia.round(2))

    fig_bmi, ax_bmi = plt.subplots(figsize=(8, 5))

    bars_bmi = ax_bmi.bar(
        bmi_prevalencia.index.astype(str),
        bmi_prevalencia.values
    )

    ax_bmi.set_title(
        "Prevalencia de diabetes por indice de masa corporal (IMC)"
    )

    ax_bmi.set_xlabel("Rango de IMC")
    ax_bmi.set_ylabel("% con diabetes/prediabetes")

    for bar, value in zip(
        bars_bmi,
        bmi_prevalencia.values
    ):
        ax_bmi.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.1f}%",
            ha="center",
            va="bottom"
        )

    fig_bmi.savefig(
        OUT_DIR / "prevalencia_bmi.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig_bmi)

    # ------------------------------
    # PREVALENCIA POR GRUPO DE EDAD
    # ------------------------------

    age_labels = [
        "18-24",
        "25-29",
        "30-34",
        "35-39",
        "40-44",
        "45-49",
        "50-54",
        "55-59",
        "60-64",
        "65-69",
        "70-74",
        "75-79",
        "80+"
    ]

    age_prevalencia = (
        df.groupby("Age")["Diabetes_binary"]
        .mean()
        .reindex(range(1, 14))
        * 100
    )

    print("\nPrevalencia por grupo de edad (%):")
    print(age_prevalencia.round(2))

    x_age = list(range(len(age_labels)))

    fig_age, ax_age = plt.subplots(figsize=(10, 5))

    ax_age.plot(
        x_age,
        age_prevalencia.values,
        marker="o"
    )

    ax_age.fill_between(
        x_age,
        age_prevalencia.values,
        alpha=0.10
    )

    ax_age.set_xticks(x_age)
    ax_age.set_xticklabels(
        age_labels,
        rotation=45
    )

    ax_age.set_title(
        "Prevalencia de diabetes por grupo de edad"
    )

    ax_age.set_xlabel(
        "Grupo de edad (categoria BRFSS)"
    )

    ax_age.set_ylabel(
        "% con diabetes/prediabetes"
    )

    fig_age.savefig(
        OUT_DIR / "prevalencia_edad.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig_age)

    # ------------------------------------
    # CORRELACION CON LA VARIABLE OBJETIVO
    # ------------------------------------

    correlaciones = (
        df.corr(numeric_only=True)["Diabetes_binary"]
        .drop("Diabetes_binary")
        .sort_values()
    )

    print("\nCorrelacion con Diabetes_binary:")
    print(
        correlaciones
        .sort_values(ascending=False)
        .round(3)
    )

    fig_corr, ax_corr = plt.subplots(figsize=(9, 7))

    bars_corr = ax_corr.barh(
        correlaciones.index,
        correlaciones.values
    )

    ax_corr.axvline(
        0,
        linewidth=0.8
    )

    ax_corr.set_title(
        "Correlacion de variables con Diabetes_binary"
    )

    ax_corr.set_xlabel(
        "Correlacion de Pearson"
    )

    for bar, value in zip(
        bars_corr,
        correlaciones.values
    ):
        ax_corr.text(
            value,
            bar.get_y() + bar.get_height() / 2,
            f" {value:.2f}",
            va="center"
        )

    fig_corr.savefig(
        OUT_DIR / "correlacion_variables.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig_corr)

    # ##fin##


if __name__ == "__main__":
    main()
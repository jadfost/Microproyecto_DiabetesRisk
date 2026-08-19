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
    print(df["Diabetes_binary"].value_counts(normalize=True))

    counts = df["Diabetes_binary"].value_counts().sort_index()
    fig, ax = plt.subplots()
    ax.bar(["Sin diabetes", "Prediabetes/diabetes"], counts.values)
    ax.set_title("Balance de clases")
    fig.savefig(OUT_DIR / "balance_clases.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()

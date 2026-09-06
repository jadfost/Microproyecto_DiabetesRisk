"""
Entrenamiento de modelos DiabetesRisk (Entrega 2).

Entrena y compara varios modelos supervisados para predecir riesgo de
diabetes/prediabetes a partir de variables demográficas y de estilo de vida,
tratando el desbalance de clases (86% / 14%).

Registra cada corrida (parámetros, métricas y el modelo serializado) en MLflow.
Ejecutar con un MLflow tracking server activo (local o en la instancia EC2):

    mlflow server --host 0.0.0.0 --port 5000
    export MLFLOW_TRACKING_URI=http://<IP_O_LOCALHOST>:5000
    python src/train.py
"""
import json
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)
from imblearn.over_sampling import SMOTE

DATA_PATH = Path("data/diabetes_binary_health_indicators_BRFSS2015.csv")
MODEL_OUT = Path("src/model.pkl")
METRICS_OUT = Path("src/metrics.json")
FEATURES = [
    "HighBP", "HighChol", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "GenHlth",
    "DiffWalk", "Age", "Income",
]
TARGET = "Diabetes_binary"

mlflow.set_experiment("diabetesrisk")


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def evaluate(model, X_test, y_test, scaler=None):
    X_eval = scaler.transform(X_test) if scaler is not None else X_test
    y_pred = model.predict(X_eval)
    y_proba = model.predict_proba(X_eval)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    cm = confusion_matrix(y_test, y_pred)
    return metrics, cm


def run_experiment(name, model, X_train, y_train, X_test, y_test, scaler=None, use_smote=False):
    with mlflow.start_run(run_name=name):
        Xtr, ytr = X_train, y_train
        if use_smote:
            Xtr, ytr = SMOTE(random_state=42).fit_resample(Xtr, ytr)
            mlflow.log_param("smote", True)
        else:
            mlflow.log_param("smote", False)

        Xtr_fit = scaler.fit_transform(Xtr) if scaler is not None else Xtr
        model.fit(Xtr_fit, ytr)

        metrics, cm = evaluate(model, X_test, y_test, scaler)

        mlflow.log_param("model_type", type(model).__name__)
        mlflow.log_param("features", ",".join(FEATURES))
        mlflow.log_param("n_features", len(FEATURES))
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        print(f"\n=== {name} ===")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        print("Matriz de confusión:\n", cm)

        return model, metrics, cm


def main():
    X_train, X_test, y_train, y_test = load_data()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Prevalencia en train: {y_train.mean():.3f}")

    results = {}
    fitted_models = {}

    # 1. Regresión Logística (baseline), sin balanceo
    scaler1 = StandardScaler()
    model1 = LogisticRegression(max_iter=1000, random_state=42)
    fit1, m1, cm1 = run_experiment("logreg_baseline", model1, X_train, y_train, X_test, y_test, scaler=scaler1)
    results["logreg_baseline"] = {**m1, "confusion_matrix": cm1.tolist()}
    fitted_models["logreg_baseline"] = fit1

    # 2. Regresión Logística + SMOTE (maneja el desbalance)
    scaler2 = StandardScaler()
    model2 = LogisticRegression(max_iter=1000, random_state=42)
    fit2, m2, cm2 = run_experiment("logreg_smote", model2, X_train, y_train, X_test, y_test, scaler=scaler2, use_smote=True)
    results["logreg_smote"] = {**m2, "confusion_matrix": cm2.tolist()}
    fitted_models["logreg_smote"] = fit2

    # 3. Random Forest con class_weight balanceado
    model3 = RandomForestClassifier(
        n_estimators=200, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1
    )
    fit3, m3, cm3 = run_experiment(
        "random_forest_balanced", model3, X_train, y_train, X_test, y_test, scaler=None
    )
    results["random_forest_balanced"] = {**m3, "confusion_matrix": cm3.tolist()}
    fitted_models["random_forest_balanced"] = fit3

    # 4. Random Forest + SMOTE
    model4 = RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
    )
    fit4, m4, cm4 = run_experiment(
        "random_forest_smote", model4, X_train, y_train, X_test, y_test, scaler=None, use_smote=True
    )
    results["random_forest_smote"] = {**m4, "confusion_matrix": cm4.tolist()}
    fitted_models["random_forest_smote"] = fit4
       # 5. Gradient Boosting + SMOTE
    model5 = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
    )

    fit5, m5, cm5 = run_experiment(
        "gradient_boosting_smote",
        model5,
        X_train,
        y_train,
        X_test,
        y_test,
        scaler=None,
        use_smote=True,
    )

    results["gradient_boosting_smote"] = {
        **m5,
        "confusion_matrix": cm5.tolist(),
    }

    fitted_models["gradient_boosting_smote"] = fit5
     ###
    # Selección: mejor modelo por recall (prioridad clínica: no perder casos positivos)
    best_name = max(results, key=lambda k: results[k]["recall"])
    best_model = fitted_models[best_name]
    print(f"\nMejor modelo por recall: {best_name} -> {results[best_name]}")

    Path("src").mkdir(exist_ok=True)
    joblib.dump({"model": best_model, "features": FEATURES, "model_name": best_name}, MODEL_OUT)
    with open(METRICS_OUT, "w") as f:
        json.dump({"results": results, "selected_model": best_name, "features": FEATURES}, f, indent=2)

    print(f"\nModelo guardado en {MODEL_OUT} (modelo seleccionado: {best_name})")
    print(f"Métricas guardadas en {METRICS_OUT}")


if __name__ == "__main__":
    main()
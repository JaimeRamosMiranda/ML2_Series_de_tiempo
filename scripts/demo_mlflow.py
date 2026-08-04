# Demo de MLflow — aprender a usar Tracking + Registry + pyfunc.
#
# Objetivo: enseñar el flujo completo de MLflow con un ejemplo simple
# (AirPassengers), antes de aplicarlo al proyecto real.
#
# Ejecutar:  python scripts/demo_mlflow.py

import os
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

# ----------------------------------------------------------------------------
# 0) CONFIGURACIÓN
# ----------------------------------------------------------------------------
# "mlruns" = carpeta local donde MLflow guarda experimentos, runs y artefactos.
# (Ya está en .gitignore: los artefactos NO van al repositorio git.)
# Tracking local con backend SQLite (recomendado por MLflow 3.x).
# La BD guarda experimentos/runs; los artefactos van a la carpeta mlruns/.
mlflow.set_tracking_uri("sqlite:///" + str((Path(__file__).resolve().parents[1] / "mlruns" / "mlflow.db")).replace("\\", "/"))

# Crear/abrir un "experimento". Un experimento agrupa varios "runs" comparables.
EXPERIMENT_NAME = "demo_airpassengers"
mlflow.set_experiment(EXPERIMENT_NAME)

DATA_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "AirPassengers.csv"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    """Carga la serie y crea las features de lags que usará el modelo."""
    df = pd.read_csv(DATA_PATH)
    df["Month"] = pd.to_datetime(df["Month"])
    df = df.set_index("Month").sort_index()
    df.columns = ["y"]

    # Características supervisadas (el modelo aprende a predecir "y" a partir de):
    df["month"] = df.index.month        # estacionalidad: mes del año
    df["lag12"] = df["y"].shift(12)     # mismo mes del año anterior
    df["lag1"] = df["y"].shift(1)       # valor del mes anterior
    return df.dropna()


def train():
    df = load_data()
    features = ["month", "lag12", "lag1"]

    # Split temporal (NO aleatorio, para no romper el orden de la serie):
    # entrenamos con todo excepto los últimos 12 meses, que son el test/validación.
    train_df = df.iloc[:-12]
    test_df = df.iloc[-12:]

    X_train, y_train = train_df[features], train_df["y"]
    X_test, y_test = test_df[features], test_df["y"]

    # 1) Parámetros del modelo: los registramos en MLflow para reproducibilidad.
    params = {"n_estimators": 200, "max_depth": 6, "random_state": 42}
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    # 2) Métricas: definen "qué tan bueno" es el modelo.
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    mae = float(mean_absolute_error(y_test, pred))
    mape = float(mean_absolute_percentage_error(y_test, pred))

    # Gráfica y predicciones como "artefactos" (evidencia visual del run).
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(test_df.index, y_test, label="Real", marker="o")
    ax.plot(test_df.index, pred, label="Predicción", marker="x")
    ax.set_title("AirPassengers — últimos 12 meses (test)")
    ax.legend()
    plot_path = OUTPUT_DIR / "demo_forecast.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    pred_df = pd.DataFrame({"Month": test_df.index, "real": y_test, "pred": pred})
    pred_path = OUTPUT_DIR / "demo_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    # ----------------------------------------------------------------------
    # 3) EL RUN DE MLFLOW: agrupa params + metrics + artifacts en un solo run.
    # ----------------------------------------------------------------------
    with mlflow.start_run(run_name="random_forest_v1"):
        mlflow.log_params(params)                 # parámetros
        mlflow.log_metrics({"rmse": rmse, "mae": mae, "mape": mape})  # métricas
        mlflow.log_artifact(str(plot_path))       # artefactos
        mlflow.log_artifact(str(pred_path))
        mlflow.set_tag("model_family", "random_forest")
        mlflow.set_tag("dataset", "airpassengers")

        # 4) Guardar el modelo DENTRO del run (con flavor sklearn → pyfunc).
        #    "model" es la ruta del artefacto dentro del run.
        mlflow.sklearn.log_model(sk_model=model, name="model")

        run_id = mlflow.active_run().info.run_id
        run_uri = f"runs:/{run_id}/model"

    # 5) REGISTRAR EL MODELO en el Model Registry (para versionarlo y poder
    #    cargarlo después por nombre, no por run). La primera vez se crea.
    registered = mlflow.register_model(model_uri=run_uri, name="airpassengers_demo")
    version = registered.version

    # Alias "champion" → puntero a la mejor versión. Se carga con @champion.
    from mlflow.tracking import MlflowClient
    MlflowClient().set_registered_model_alias(
        name="airpassengers_demo", alias="champion", version=version
    )

    print(f"\nRun ID       : {run_id}")
    print(f"Experiment   : {EXPERIMENT_NAME}")
    print(f"Versión      : v{version} (alias: champion)")
    print(f"RMSE={rmse:.1f} | MAE={mae:.1f} | MAPE={mape*100:.1f}%\n")
    print("Para ver la UI local (tracking + registry + artefactos):")
    print("  python -m mlflow ui")
    print("  y abrir http://127.0.0.1:5000")


def load_and_predict():
    """Carga el modelo productivo desde el Registry y predice."""
    print("\n--- Cargando modelo productivo desde el Registry ---")
    model = mlflow.pyfunc.load_model("models:/airpassengers_demo@champion")

    example = pd.DataFrame(
        {"month": [1, 2, 3], "lag12": [360, 342, 406], "lag1": [432, 360, 342]}
    )
    pred = model.predict(example)
    print("Predicciones para 3 meses ejemplo:", pred)


if __name__ == "__main__":
    train()
    load_and_predict()

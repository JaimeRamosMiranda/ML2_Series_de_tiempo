#!/usr/bin/env python
"""Sube la evidencia del proyecto a MLflow remoto en DagsHub (Fase 4/6).

El entregable exige un **link público de MLflow** con experimentos que tengan
parámetros, métricas y artefactos, y un modelo productivo con alias. Este script
re-registra en DagsHub la evidencia local del proyecto:

- Experimento ``demand_forecast_fase3``: run ``lightgbm_production`` con las
  métricas del modelo de producción, sus artefactos (predicciones y gráfica) y el
  modelo registrado con el alias ``Production``.
- Experimento ``demand_forecast_insights``: los runs del agente genAI generados
  con el LLM de Groq (contexto, series similares e insight como artefactos).

Requiere en ``.env``: ``DAGSHUB_OWNER``, ``DAGSHUB_REPO`` y ``DAGSHUB_TOKEN``.

Ejemplos:
    python scripts/upload_dagshub_evidence.py          # evidencia completa
    python scripts/upload_dagshub_evidence.py --only-model
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import mlflow
from dotenv import dotenv_values
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[1]

LOCAL_URI = f"sqlite:///{(ROOT / 'mlruns' / 'mlflow.db').as_posix()}"
FASE3_EXP = "demand_forecast_fase3"
INSIGHTS_EXP = "demand_forecast_insights"
MODEL_NAME = "demand_forecast"
PROD_RUN_ID = "0ca4a1eff4934430ae13392505fe382e"  # run lightgbm (producción local)
MODEL_DIR = ROOT / "models" / "demand_forecast"


def _artifact_dir(run) -> Path:
    """Directorio de artefactos de un run local (desde su artifact_uri)."""
    uri = run.info.artifact_uri
    return Path(uri.removeprefix("file:")) if uri.startswith("file:") else ROOT / "mlruns"


def _wait_model_sync(client: MlflowClient, run_id: str, timeout_s: int = 120) -> bool:
    """Espera a que el directorio ``model`` del run remoto aparezca.

    DagsHub sincroniza los artefactos de forma asíncrona: si se registra el
    modelo antes de que el directorio ``model`` exista en el run, MLflow
    devuelve "Unable to find a logged_model". Se espera hasta que aparezca.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if client.list_artifacts(run_id, "model"):
            return True
        time.sleep(3)
    return False


def upload_model_run() -> None:
    """Sube el run de producción y registra el modelo con alias ``Production``."""
    local_client = MlflowClient(LOCAL_URI)
    run = local_client.get_run(PROD_RUN_ID)
    metrics, params = run.data.metrics, run.data.params
    artifacts = sorted(p for p in _artifact_dir(run).iterdir() if p.is_file())

    mlflow.set_experiment(FASE3_EXP)
    with mlflow.start_run(run_name="lightgbm_production"):
        mlflow.log_params(params)
        mlflow.log_metrics({
            "mase": metrics["mase"],
            "wape": metrics["wape"],
            "rel_bias_pct": metrics["rel_bias_pct"],
            "bias": metrics.get("bias", 0.0),
        })
        for art in artifacts:
            mlflow.log_artifact(str(art))
        # DagsHub no persiste el directorio `model` creado por
        # mlflow.sklearn.log_model; se suben los archivos del modelo exportado
        # como artefactos bajo `model/` para que `runs:/<id>/model` resuelva.
        for f in sorted(MODEL_DIR.iterdir()):
            mlflow.log_artifact(str(f), artifact_path="model")
        remote_run_id = mlflow.active_run().info.run_id

    client = MlflowClient()
    if not _wait_model_sync(client, remote_run_id):
        print("  [aviso] el directorio model no apareció en el run remoto.")
    registered = mlflow.register_model(
        model_uri=f"runs:/{remote_run_id}/model", name=MODEL_NAME
    )
    for _ in range(40):
        if client.get_model_version(MODEL_NAME, registered.version).status == "READY":
            break
        time.sleep(0.5)
    client.set_registered_model_alias(MODEL_NAME, "Production", registered.version)
    print(f"  Modelo {MODEL_NAME} v{registered.version} -> alias Production")


def upload_insights() -> None:
    """Sube los runs del agente genAI generados con el LLM (experimento insights)."""
    mlflow.set_experiment(INSIGHTS_EXP)
    local_client = MlflowClient(LOCAL_URI)
    exp = local_client.get_experiment_by_name(INSIGHTS_EXP)
    if exp is None:
        print("  Sin experimento local de insights; se omite.")
        return
    runs = local_client.search_runs([exp.experiment_id])
    for run in runs:
        if run.data.params.get("generator") != "llm_groq":
            continue
        art_dir = _artifact_dir(run)
        with mlflow.start_run(run_name=run.data.tags.get("mlflow.runName", "insight")):
            mlflow.log_params(run.data.params)
            mlflow.log_metrics({
                k: v for k, v in run.data.metrics.items()
                if k.startswith("latency") or k in ("prompt_tokens", "completion_tokens", "total_tokens", "references_series")
            })
            for art in sorted(art_dir.iterdir()) if art_dir.exists() else []:
                if art.is_file():
                    mlflow.log_artifact(str(art))
        print(f"  Insight de tienda {run.data.params.get('store')}, "
              f"artículo {run.data.params.get('item')} subido.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only-model", action="store_true",
                        help="Subir solo el run de producción + modelo.")
    args = parser.parse_args()

    env = dotenv_values(ROOT / ".env")
    owner = env.get("DAGSHUB_OWNER", "jaimeramos124")
    repo = env.get("DAGSHUB_REPO", "")
    token = env.get("DAGSHUB_TOKEN", "")
    if not token or not repo:
        print("Falta DAGSHUB_TOKEN/DAGSHUB_REPO en .env. Abortando.")
        sys.exit(1)

    import dagshub

    os.environ["MLFLOW_TRACKING_USERNAME"] = owner
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    dagshub.init(repo, owner, mlflow=True)

    print("Subiendo evidencia a DagsHub...")
    upload_model_run()
    if not args.only_model:
        upload_insights()

    url = f"https://dagshub.com/{owner}/{repo}/experiments"
    print("\nEvidencia: " + url)


if __name__ == "__main__":
    main()

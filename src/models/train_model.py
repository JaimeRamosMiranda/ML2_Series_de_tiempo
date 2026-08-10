"""Runner de entrenamiento + evaluación con registro en MLflow.

Cada configuración de modelo se entrena, se evalúa sobre el holdout y se
registra como un *run* de MLflow con:
    - parámetros del modelo,
    - todas las métricas (incl. ``bias`` y ``rel_bias_pct``),
    - artefactos: CSV de predicciones y gráfica del pronóstico agregado,
    - el modelo en el Model Registry (``mlflow.pyfunc`` / ``sklearn``).

Después de cada run se invoca ``promote_best_model`` para mover el alias
``Production`` a la mejor versión según la regla de decisión:
    ranking combinado MASE + WAPE, descartando modelos con |sesgo relativo|
    mayor a ``MAX_REL_BIAS_PCT`` y desempatando por el menor sesgo absoluto.
"""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.base import BaseEstimator, RegressorMixin

matplotlib.use("Agg")

from src.models.metrics import compute_all_metrics  # noqa: E402

# Regla de decisión: sesgo relativo máximo aceptable para ser candidato a
# Production. Fuera de este rango el modelo se descarta (sub/sobreestimación).
MAX_REL_BIAS_PCT = 5.0


# ---------------------------------------------------------------------------
# Baselines como estimadores compatibles con sklearn (usan columnas de features)
# ---------------------------------------------------------------------------
class NaiveRegressor(BaseEstimator, RegressorMixin):
    """Baseline naive: predice el valor del día anterior (lag_1)."""

    def __init__(self, lag_col: str = "lag_1"):
        self.lag_col = lag_col

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.asarray(X[self.lag_col], dtype=float)


class SeasonalNaiveRegressor(NaiveRegressor):
    """Baseline seasonal-naive: predice el valor de hace 7 días (lag_7)."""

    def __init__(self, lag_col: str = "lag_7"):
        super().__init__(lag_col=lag_col)


class MeanRegressor(BaseEstimator, RegressorMixin):
    """Baseline media por serie: predice la media móvil de 30 días."""

    def __init__(self, mean_col: str = "rolling_mean_30"):
        self.mean_col = mean_col

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.asarray(X[self.mean_col], dtype=float)


@dataclass
class ModelConfig:
    """Configuración de un modelo a probar (registrada como run de MLflow)."""

    name: str
    family: str
    estimator_factory: callable  # fábrica que devuelve un estimador nuevo
    params: dict


# ---------------------------------------------------------------------------
# Entrenamiento + registro
# ---------------------------------------------------------------------------
def _log_sklearn_model(estimator) -> None:
    """Registra el modelo en el run, reintentando con los tipos whitelistados.

    MLflow 3.x serializa los estimadores sklearn con skops, que exige
    ``skops_trusted_types`` para clases que no están en la lista por defecto
    (p.ej. LGBMRegressor o los baselines custom). Se extraen los tipos no
    confiables del propio error y se reintenta automáticamente.
    """
    try:
        mlflow.sklearn.log_model(sk_model=estimator, name="model")
    except Exception as exc:  # noqa: BLE001
        match = re.search(r"Untrusted types found in the file: \[([^\]]*)\]", str(exc))
        if not match:
            raise
        trusted = [t.strip().strip("'") for t in match.group(1).split(",")]
        mlflow.sklearn.log_model(
            sk_model=estimator, name="model", skops_trusted_types=trusted
        )


def _save_predictions_artifact(
    df_test: pd.DataFrame,
    y_pred: np.ndarray,
) -> Path:
    """Guarda un CSV con real vs predicho por (store, item, date)."""
    out = pd.DataFrame({
        "date": df_test["date"].values,
        "store": df_test["store"].values,
        "item": df_test["item"].values,
        "sales": df_test["sales"].values,
        "pred": y_pred,
    })
    tmp = Path(tempfile.mkdtemp()) / "predictions.csv"
    out.to_csv(tmp, index=False)
    return tmp


def _save_forecast_plot(
    df_test: pd.DataFrame,
    y_pred: np.ndarray,
    model_name: str,
) -> Path:
    """Gráfica del pronóstico agregado diario (real vs predicho)."""
    agg = df_test.assign(pred=y_pred).groupby("date")[["sales", "pred"]].sum()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(agg.index, agg["sales"], label="Real", marker="o", ms=3, lw=1.2)
    ax.plot(agg.index, agg["pred"], label="Predicción", marker="x", ms=3, lw=1.2)
    ax.set_title(f"Pronóstico agregado diario — {model_name}")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Unidades")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    tmp = Path(tempfile.mkdtemp()) / "forecast_plot.png"
    fig.savefig(tmp, dpi=150)
    plt.close(fig)
    return tmp


def train_and_log_model(
    cfg: ModelConfig,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    df_test: pd.DataFrame,
    series_ids_test: pd.Series,
    naive_scale: pd.Series,
    experiment_name: str,
    registered_model_name: str = "demand_forecast",
    feature_names: list[str] | None = None,
) -> dict:
    """Entrena un modelo, lo evalúa y registra el run + modelo en MLflow.

    Returns:
        Dict con run_id, métricas, versión registrada y si fue promovido a
        Production.
    """
    estimator = cfg.estimator_factory()
    estimator.fit(X_train[feature_names] if feature_names else X_train, y_train)

    X_test_model = X_test[feature_names] if feature_names else X_test
    y_pred = estimator.predict(X_test_model)

    metrics = compute_all_metrics(
        y_test.values, y_pred, series_ids=series_ids_test, naive_scale=naive_scale
    )

    pred_csv = _save_predictions_artifact(df_test, y_pred)
    plot_png = _save_forecast_plot(df_test, y_pred, cfg.name)

    with mlflow.start_run(run_name=cfg.name):
        mlflow.log_params(cfg.params)
        mlflow.log_metrics(metrics)
        mlflow.set_tags({
            "model_family": cfg.family,
            "dataset": "store_item_demand",
            "split": "train<=2017-09-30 | holdout=2017-10-01..2017-12-31",
        })
        mlflow.log_artifact(str(pred_csv))
        mlflow.log_artifact(str(plot_png))

        try:
            _log_sklearn_model(estimator)
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
            logged = mlflow.register_model(
                model_uri=model_uri, name=registered_model_name
            )
            mlflow.log_params({
                "registered_model_name": registered_model_name,
                "registered_model_version": str(logged.version),
            })
            registered_version = logged.version
        except Exception as exc:  # noqa: BLE001 - no bloquear el run
            mlflow.log_params({
                "registered_model_name": registered_model_name,
                "registered_model_error": str(exc)[:500],
            })
            registered_version = None

        run_id = mlflow.active_run().info.run_id
        result = {"run_id": run_id, "metrics": metrics, "version": registered_version}

    promoted = False
    if registered_version is not None:
        promoted = promote_best_model(
            experiment_name, registered_model_name, alias="Production"
        )
        result["promoted_to_production"] = promoted

    print(
        f"[{cfg.name}] MASE={metrics['mase']:.3f} | WAPE={metrics['wape']:.2f}% "
        f"| bias={metrics['bias']:+.2f} | rel_bias={metrics['rel_bias_pct']:+.2f}% "
        f"| run={run_id[:8]}" + (f" | v{registered_version}" if registered_version else "")
    )
    return result


# ---------------------------------------------------------------------------
# Selección del mejor modelo (regla MASE + WAPE con filtro de sesgo)
# ---------------------------------------------------------------------------
def _wait_ready(client: MlflowClient, name: str, version: int, timeout_s: int = 20) -> bool:
    """Espera a que una versión registrada esté READY antes de asignar alias.

    ``register_model`` crea la versión en estado PENDING mientras sube los
    artefactos; asignar un alias antes de que esté READY puede no persistir.
    """
    import time

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        mv = client.get_model_version(name, version)
        if mv.status == "READY":
            return True
        time.sleep(0.5)
    return False


def promote_best_model(
    experiment_name: str,
    registered_model_name: str,
    alias: str = "Production",
    max_rel_bias: float = MAX_REL_BIAS_PCT,
) -> bool:
    """Mueve el alias ``alias`` a la mejor versión registrada del experimento.

    Ranking: se descartan los runs con ``|rel_bias_pct| > max_rel_bias`` y se
    ordena el resto por rango combinado de MASE y WAPE; desempate por menor
    sesgo absoluto. Si no queda candidato, se usa el mejor por WAPE.
    """
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        return False

    runs = client.search_runs(experiment_ids=[exp.experiment_id])
    rows = []
    for r in runs:
        m = r.data.metrics
        p = r.data.params
        version = p.get("registered_model_version")
        if version is None or "mase" not in m or "wape" not in m:
            continue
        rows.append({
            "run_id": r.info.run_id,
            "version": int(version),
            "mase": m["mase"],
            "wape": m["wape"],
            "rel_bias_pct": m.get("rel_bias_pct", 0.0),
            "name": r.data.tags.get("mlflow.runName", r.info.run_id[:8]),
        })
    if not rows:
        return False

    candidates = [r for r in rows if abs(r["rel_bias_pct"]) <= max_rel_bias]
    pool = candidates if candidates else rows

    by_mase = {r["run_id"]: i for i, r in enumerate(sorted(pool, key=lambda r: r["mase"]))}
    by_wape = {r["run_id"]: i for i, r in enumerate(sorted(pool, key=lambda r: r["wape"]))}
    winner = min(
        pool,
        key=lambda r: (by_mase[r["run_id"]] + by_wape[r["run_id"]], abs(r["rel_bias_pct"])),
    )

    _wait_ready(client, registered_model_name, winner["version"])
    client.set_registered_model_alias(
        name=registered_model_name, alias=alias, version=winner["version"]
    )
    print(
        f"  -> Production actualizado: {winner['name']} (v{winner['version']}) "
        f"MASE={winner['mase']:.3f} WAPE={winner['wape']:.2f}% "
        f"rel_bias={winner['rel_bias_pct']:+.2f}%"
    )
    return True

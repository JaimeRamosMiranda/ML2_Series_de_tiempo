"""Métricas de evaluación para el pronóstico de demanda multi-series.

Incluye las métricas de error estándar (RMSE, MAE, MAPE, sMAPE, MASE, WAPE,
MedAE, Max Error, R²) y el **sesgo (bias)** como métrica de decisión.

Convención de sesgo:
    bias = mean(pred - y_true)

- ``bias < 0``  → el modelo **subestima** sistemáticamente (riesgo de rotura de
  stock en producción, el problema clásico que solo RMSE/MAE no detectan).
- ``bias > 0``  → el modelo **sobreestima** (riesgo de exceso de inventario).
- ``rel_bias_pct = bias / mean(y_true) * 100`` → magnitud relativa del sesgo,
  comparable entre series y reportable en MLflow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

EPSILON = 1e-8


# ---------------------------------------------------------------------------
# Métricas individuales
# ---------------------------------------------------------------------------
def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Sesgo medio: ``mean(pred - y_true)``. Negativo ⇒ subestima."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(y_pred - y_true))


def relative_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Sesgo relativo en porcentaje respecto a la media real.

    ``rel_bias_pct = bias / mean(y_true) * 100``. Un modelo que subestima un
    3% tendrá ``rel_bias_pct = -3.0``.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = float(np.mean(y_true))
    return float(np.mean(y_pred - y_true) / denom * 100.0) if denom else 0.0


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error (%)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred) + EPSILON
    return float(np.mean(2.0 * np.abs(y_true - y_pred) / denom) * 100.0)


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Absolute Percentage Error (%). Visión de negocio/inventario."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = float(np.sum(np.abs(y_true)))
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100.0) if denom else 0.0


def naive_scale_by_series(
    df: pd.DataFrame,
    target: str = "sales",
    group_cols: tuple[str, str] = ("store", "item"),
    date_col: str = "date",
) -> pd.Series:
    """Escala naive por serie: ``mean(|y_t - y_{t-1}|)`` sobre el entrenamiento.

    Es el denominador del MASE. Se calcula sobre el split de entrenamiento
    (historial real) y se reutiliza para evaluar las predicciones del holdout.
    """
    out = df.sort_values(list(group_cols) + [date_col])
    scales = (
        out.groupby(list(group_cols))[target]
        .apply(lambda s: float(s.diff().abs().mean()))
    )
    return scales


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    series_ids: pd.Series | list,
    naive_scale: pd.Series,
) -> float:
    """Mean Absolute Scaled Error (MASE) agregado sobre las series.

    Por cada serie: ``MASE_s = MAE_s / naive_scale_s`` y se promedian los MASE
    de todas las series. ``MASE < 1`` indica que el modelo supera al naive.
    Las series con escala naive cero (constantes) se excluyen del promedio.
    """
    df = pd.DataFrame(
        {"y_true": np.asarray(y_true, dtype=float),
         "y_pred": np.asarray(y_pred, dtype=float),
         "series": list(series_ids)}
    )

    df["abs_err"] = (df["y_true"] - df["y_pred"]).abs()
    mae_by_series = df.groupby("series")["abs_err"].mean()
    scale_aligned = mae_by_series.index.to_series().map(naive_scale).astype(float)
    valid = scale_aligned > EPSILON
    mase_by_series = mae_by_series[valid] / scale_aligned[valid]
    if len(mase_by_series) == 0:
        return float("nan")
    return float(mase_by_series.mean())


# ---------------------------------------------------------------------------
# Métricas agregadas
# ---------------------------------------------------------------------------
def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    series_ids: pd.Series | list | None = None,
    naive_scale: pd.Series | None = None,
) -> dict[str, float]:
    """Calcula todas las métricas del proyecto en un solo dict.

    Args:
        y_true: valores reales del holdout.
        y_pred: predicciones del modelo.
        series_ids: (store, item) por fila; requerido para MASE.
        naive_scale: escala naive por serie (de ``naive_scale_by_series``);
            requerido para MASE.

    Returns:
        Dict con ``rmse``, ``mae``, ``mape``, ``smape``, ``wape``, ``medae``,
        ``max_error``, ``r2``, ``bias``, ``rel_bias_pct`` y, si hay series,
        ``mase``.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred) * 100.0),
        "smape": smape(y_true, y_pred),
        "wape": wape(y_true, y_pred),
        "medae": float(np.median(np.abs(y_true - y_pred))),
        "max_error": float(max_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "bias": bias(y_true, y_pred),
        "rel_bias_pct": relative_bias(y_true, y_pred),
    }

    if series_ids is not None and naive_scale is not None:
        metrics["mase"] = mase(y_true, y_pred, series_ids, naive_scale)
    return metrics

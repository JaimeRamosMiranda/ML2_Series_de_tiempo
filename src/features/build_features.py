"""Construcción de features para el pronóstico de demanda multi-series.

El pipeline convierte las ventas diarias (store, item, date, sales) en un dataset
supervisado de regresión: cada fila contiene features (calendario, lags y medias/
desviaciones móviles) que permiten predecir ``sales`` del día.

Nota sobre fuga de datos: los lags y las ventanas móviles usan únicamente valores
*pasados* (shift(1) antes del rolling), de modo que ningún feature usa información
del día que se está prediciendo.
"""

from __future__ import annotations

import pandas as pd

DEFAULT_GROUP_COLS = ("store", "item")
DEFAULT_LAGS = (1, 7, 30)
DEFAULT_WINDOWS = (7, 30)
CALENDAR_COLUMNS = ["year", "month", "day", "dayofweek", "weekofyear", "dayofyear"]


def add_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Agrega features de calendario derivadas de una columna datetime.

    Args:
        df: DataFrame con una columna de fecha.
        date_col: nombre de la columna datetime.

    Returns:
        DataFrame con las columnas year, month, day, dayofweek, weekofyear y dayofyear.
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out[date_col])
    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month
    out["day"] = out["date"].dt.day
    out["dayofweek"] = out["date"].dt.dayofweek
    out["weekofyear"] = out["date"].dt.isocalendar().week.astype("int64")
    out["dayofyear"] = out["date"].dt.dayofyear
    return out


def add_lag_features(
    df: pd.DataFrame,
    target: str = "sales",
    lags: tuple[int, ...] = DEFAULT_LAGS,
    group_cols: tuple[str, ...] = DEFAULT_GROUP_COLS,
    date_col: str = "date",
) -> pd.DataFrame:
    """Agrega lags del target calculados dentro de cada serie.

    Cada serie (store, item) se ordena por fecha y se agrega el valor de la
    variable objetivo de ``lag`` días atrás.

    Args:
        df: DataFrame con las series.
        target: columna a usar como objetivo.
        lags: días de rezago a incluir (ej. (1, 7, 30)).
        group_cols: columnas que identifican cada serie.
        date_col: columna de fecha usada para ordenar.

    Returns:
        DataFrame con las columnas ``lag_{lag}`` por cada lag en ``lags``.
    """
    out = df.copy()
    out = out.sort_values(list(group_cols) + [date_col])
    for lag in lags:
        out[f"lag_{lag}"] = out.groupby(list(group_cols))[target].shift(lag)
    return out


def add_rolling_features(
    df: pd.DataFrame,
    target: str = "sales",
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
    group_cols: tuple[str, ...] = DEFAULT_GROUP_COLS,
    date_col: str = "date",
) -> pd.DataFrame:
    """Agrega medias y desviaciones móviles del target dentro de cada serie.

    Se aplica ``shift(1)`` antes del rolling para usar únicamente el historial
    pasado y evitar fuga de datos.

    Args:
        df: DataFrame con las series.
        target: columna objetivo.
        windows: ventanas (en días) para las estadísticas móviles.
        group_cols: columnas que identifican cada serie.
        date_col: columna de fecha usada para ordenar.

    Returns:
        DataFrame con columnas ``rolling_mean_{w}`` y ``rolling_std_{w}``.
    """
    out = df.copy()
    out = out.sort_values(list(group_cols) + [date_col])
    grouped = out.groupby(list(group_cols))[target]
    for w in windows:
        out[f"rolling_mean_{w}"] = grouped.transform(lambda s: s.shift(1).rolling(w).mean())
        out[f"rolling_std_{w}"] = grouped.transform(lambda s: s.shift(1).rolling(w).std())
    return out


def build_features(df: pd.DataFrame, target: str = "sales") -> pd.DataFrame:
    """Pipeline completo de features (calendario + lags + rolling).

    Args:
        df: DataFrame con las columnas date, store, item y el target.
        target: columna objetivo.

    Returns:
        DataFrame con todas las features agregadas (puede contener NaN en las
        primeras filas de cada serie, que deben eliminarse antes de modelar).
    """
    out = add_calendar_features(df, date_col="date")
    out = add_lag_features(out, target=target)
    out = add_rolling_features(out, target=target)
    return out


FEATURE_COLUMNS = (
    list(CALENDAR_COLUMNS) + ["store", "item"]
    + [f"lag_{lag}" for lag in DEFAULT_LAGS]
    + [f"rolling_mean_{w}" for w in DEFAULT_WINDOWS]
    + [f"rolling_std_{w}" for w in DEFAULT_WINDOWS]
)

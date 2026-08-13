"""Backtesting walk-forward (métricas *online*) para el pronóstico multi-series.

Simula producción sobre el **historial** (hasta 2017-09-30): divide el pasado
en ventanas de ~90 días y, para cada ventana, entrena con todo lo anterior y
predice la ventana. Así se obtienen métricas online que complementan la
evaluación offline del holdout (oct-dic 2017).

Sin fuga de datos:
    - Las features se construyen sobre el historial completo ANTES de las
      ventanas (los lags/rolling solo usan valores pasados por fila).
    - Por cada fold: ``train = fecha <= train_end``, ``test = (train_end, test_end]``.
    - El denominador del MASE (escala naive) se calcula sobre el train de cada fold.
"""

from __future__ import annotations

import pandas as pd

from src.models.metrics import compute_all_metrics, naive_scale_by_series
from src.models.train_model import ModelConfig

DEFAULT_TEST_ENDS = ["2016-12-31", "2017-03-31", "2017-06-30", "2017-09-30"]
DEFAULT_HORIZON_DAYS = 90


def walk_forward_folds(
    test_ends: list[str] = DEFAULT_TEST_ENDS,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Genera pares ``(train_end, test_end)`` para las ventanas del backtest.

    Args:
        test_ends: fecha final (inclusive) de cada ventana de test.
        horizon_days: largo aproximado de la ventana de test (en días).

    Returns:
        Lista de tuplas (train_end, test_end). El train usa ``<= train_end``
        y el test usa ``(train_end, test_end]``.
    """
    folds = []
    for test_end in test_ends:
        te = pd.Timestamp(test_end)
        ts = te - pd.Timedelta(days=horizon_days - 1)
        folds.append((ts - pd.Timedelta(days=1), te))
    return folds


def walk_forward_backtest(
    featured: pd.DataFrame,
    configs: dict[str, ModelConfig],
    feature_names: list[str],
    test_ends: list[str] = DEFAULT_TEST_ENDS,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    target: str = "sales",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ejecuta el backtest walk-forward para cada modelo.

    Args:
        featured: features construidas sobre el historial completo (con ``date``
            datetime, ``store``, ``item`` y el target).
        configs: catálogo de modelos (:class:`ModelConfig`).
        feature_names: columnas de features a usar.
        test_ends: finales de cada ventana de test.
        horizon_days: largo de la ventana de test.
        target: columna objetivo.

    Returns:
        ``(folds, agg, pooled)``:
        - ``folds``: una fila por (modelo, fold) con todas las métricas.
        - ``agg``: una fila por modelo con el promedio de las métricas entre folds.
        - ``pooled``: dict ``{modelo: DataFrame(date, store, item, sales, pred)}``
          con las predicciones acumuladas de todos los folds (para graficar).
    """
    folds_meta = walk_forward_folds(test_ends, horizon_days)

    records: list[dict] = []
    pooled: dict[str, list[pd.DataFrame]] = {name: [] for name in configs}
    for train_end, test_end in folds_meta:
        train = featured[featured["date"] <= train_end].dropna()
        test = featured[
            (featured["date"] > train_end) & (featured["date"] <= test_end)
        ].copy()

        X_train, y_train = train[feature_names], train[target]
        X_test, y_test = test[feature_names], test[target]

        df_test = test[["date", "store", "item", "sales"]].reset_index(drop=True)
        series_ids = pd.Series(list(zip(df_test["store"], df_test["item"])))
        naive_scale = naive_scale_by_series(
            train[["store", "item", "date", "sales"]], target=target
        )

        for name, cfg in configs.items():
            estimator = cfg.estimator_factory()
            estimator.fit(X_train, y_train)
            y_pred = estimator.predict(X_test)
            metrics = compute_all_metrics(
                y_test.values, y_pred, series_ids=series_ids, naive_scale=naive_scale
            )
            records.append({
                "modelo": name,
                "fold": str(test_end.date()),
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                **metrics,
            })
            pooled[name].append(df_test.assign(pred=y_pred))

    folds = pd.DataFrame(records)

    numeric = [
        "rmse", "mae", "mape", "smape", "mase", "wape", "medae",
        "max_error", "r2", "bias", "rel_bias_pct",
    ]
    agg = (
        folds.groupby("modelo", as_index=False)[numeric]
        .mean()
        .rename(columns=lambda c: c if c == "modelo" else f"{c}_online")
    )
    pooled = {
        name: pd.concat(parts).reset_index(drop=True)
        for name, parts in pooled.items()
    }
    return folds, agg, pooled


def backtest_to_mlflow(
    cfg: ModelConfig,
    folds: pd.DataFrame,
    agg: pd.DataFrame,
    experiment_name: str,
    n_folds: int,
    horizon_days: int,
) -> None:
    """Registra el backtest de un modelo como run de MLflow.

    Args:
        cfg: configuración del modelo (nombre, familia, params).
        folds: detalle por fold (una fila por (modelo, fold)).
        agg: agregado por modelo (columnas ``<metric>_online``).
        experiment_name: experimento de MLflow destino.
        n_folds: cantidad de folds (se loguea como parámetro).
        horizon_days: horizonte de cada fold (se loguea como parámetro).
    """
    import mlflow

    mlflow.set_experiment(experiment_name)
    model_folds = folds[folds["modelo"] == cfg.name]
    row = agg[agg["modelo"] == cfg.name].iloc[0]

    with mlflow.start_run(run_name=cfg.name):
        mlflow.log_params({
            "model_family": cfg.family,
            "n_folds": n_folds,
            "horizon_days": horizon_days,
            "split": "walk-forward (online), historial <= 2017-09-30",
            **{f"param_{k}": str(v) for k, v in cfg.params.items()},
        })
        metrics = {
            f"{col}_online": float(row[f"{col}_online"])
            for col in ["mase", "wape", "mae", "rmse", "bias", "rel_bias_pct"]
        }
        mlflow.log_metrics(metrics)
        for _, f in model_folds.iterrows():
            mlflow.log_metrics({
                "mase": float(f["mase"]),
                "wape": float(f["wape"]),
                "bias": float(f["bias"]),
            }, step=int(f["n_test"]))
        mlflow.log_text(
            model_folds.to_csv(index=False), f"backtest_{cfg.name}.csv"
        )

#!/usr/bin/env python
"""Entrenamiento y evaluación de modelos con registro en MLflow (Fase 3+4).

Cada modelo configurado se entrena sobre las features de train, se evalúa sobre
el holdout (oct-dic 2017) y se registra como run de MLflow + Model Registry.
El alias ``Production`` apunta a la mejor versión según la regla de decisión
(ranking MASE + WAPE con filtro de sesgo relativo <= 5%).

Ejemplos:
    python scripts/train.py                                 # baselines + LightGBM
    python scripts/train.py --models naive,lightgbm         # solo dos modelos
    python scripts/train.py --all                           # todas las familias
    python scripts/train.py --max-series 20                 # smoke test (20 series)

Requiere que las features ya existan (Fase 2):
    data/processed/train_features.csv
    data/processed/holdout_features.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlflow
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.build_features import FEATURE_COLUMNS  # noqa: E402
from src.models.configs import all_configs, baseline_configs, supervised_configs  # noqa: E402
from src.models.metrics import naive_scale_by_series  # noqa: E402
from src.models.train_model import (  # noqa: E402
    MAX_REL_BIAS_PCT,
    promote_best_model,
    train_and_log_model,
)

TRACKING_URI = f"sqlite:///{(ROOT / 'mlruns' / 'mlflow.db').as_posix()}"
EXPERIMENT_NAME = "demand_forecast_fase3"
REGISTERED_MODEL = "demand_forecast"

TARGET = "sales"


def load_data(
    max_series: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.DataFrame, pd.Series]:
    """Carga features de train/holdout, escala naive y df_test alineado.

    Returns:
        (X_train, X_test, y_train, y_test, naive_scale, df_test, series_ids_test)
    """
    train = pd.read_csv(ROOT / "data" / "processed" / "train_features.csv")
    holdout = pd.read_csv(ROOT / "data" / "processed" / "holdout_features.csv")

    if max_series is not None:
        selected = train[["store", "item"]].drop_duplicates().head(max_series)
        keys = set(zip(selected["store"], selected["item"]))
        train = train[train.apply(lambda r: (r["store"], r["item"]) in keys, axis=1)]
        holdout = holdout[holdout.apply(lambda r: (r["store"], r["item"]) in keys, axis=1)]

    train = train.dropna().reset_index(drop=True)
    holdout = holdout.dropna().reset_index(drop=True)

    features = [c for c in FEATURE_COLUMNS if c in train.columns]
    X_train = train[features]
    y_train = train[TARGET]
    X_test = holdout[features]
    y_test = holdout[TARGET]

    df_test = holdout[["date", "store", "item", "sales"]].reset_index(drop=True)
    series_ids_test = pd.Series(list(zip(df_test["store"], df_test["item"])))

    train_raw = pd.read_csv(ROOT / "data" / "processed" / "train.csv")
    naive_scale = naive_scale_by_series(train_raw)

    return X_train, X_test, y_train, y_test, naive_scale, df_test, series_ids_test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", type=str, default="naive,seasonal_naive,mean,lightgbm",
        help="Lista separada por comas de modelos a ejecutar.",
    )
    parser.add_argument("--all", action="store_true", help="Ejecutar todas las familias.")
    parser.add_argument("--experiment", default=EXPERIMENT_NAME)
    parser.add_argument("--max-series", type=int, default=None,
                        help="Limitar a N series (smoke test rápido).")
    args = parser.parse_args()

    if args.all:
        configs = all_configs()
    else:
        available = {**baseline_configs(), **supervised_configs()}
        configs = {
            name: available[name]
            for name in args.models.split(",")
            if name in available
        }
    if not configs:
        available = sorted(set(baseline_configs()) | set(supervised_configs()))
        print(f"Sin modelos válidos. Opciones: {', '.join(available)}")
        sys.exit(1)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(args.experiment)
    print(f"Tracking: {TRACKING_URI}")
    print(f"Experimento: {args.experiment} | modelos: {', '.join(configs)}")
    print(f"Filtro de sesgo: |rel_bias| <= {MAX_REL_BIAS_PCT}% para Production\n")

    X_train, X_test, y_train, y_test, naive_scale, df_test, series_ids_test = load_data(
        args.max_series
    )

    if len(X_test) != len(df_test):
        print(f"ADVERTENCIA: filas X_test={len(X_test)} vs df_test={len(df_test)}; "
              "se recortan las predicciones al mínimo.", file=sys.stderr)
        n = min(len(X_test), len(df_test))
        X_test, df_test = X_test.iloc[:n], df_test.iloc[:n]
        y_test, series_ids_test = y_test.iloc[:n], series_ids_test.iloc[:n]

    print(f"Train: {X_train.shape} | Holdout: {X_test.shape}\n")

    for name, cfg in configs.items():
        train_and_log_model(
            cfg,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            df_test=df_test,
            series_ids_test=series_ids_test,
            naive_scale=naive_scale,
            experiment_name=args.experiment,
            registered_model_name=REGISTERED_MODEL,
            feature_names=list(X_train.columns),
        )

    promote_best_model(args.experiment, REGISTERED_MODEL, alias="Production")
    print("\nListo. Para ver la UI:  python -m mlflow ui")


if __name__ == "__main__":
    main()

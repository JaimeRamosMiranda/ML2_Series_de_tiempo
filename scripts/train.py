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
from src.models.metrics import naive_scale_by_series  # noqa: E402
from src.models.train_model import (  # noqa: E402
    MAX_REL_BIAS_PCT,
    MeanRegressor,
    ModelConfig,
    NaiveRegressor,
    SeasonalNaiveRegressor,
    promote_best_model,
    train_and_log_model,
)

TRACKING_URI = f"sqlite:///{(ROOT / 'mlruns' / 'mlflow.db').as_posix()}"
EXPERIMENT_NAME = "demand_forecast_fase3"
REGISTERED_MODEL = "demand_forecast"

TARGET = "sales"

BASE_MODELS: dict[str, ModelConfig] = {
    "naive": ModelConfig(
        name="naive",
        family="baseline",
        estimator_factory=lambda: NaiveRegressor(),
        params={"type": "naive", "lag": 1},
    ),
    "seasonal_naive": ModelConfig(
        name="seasonal_naive",
        family="baseline",
        estimator_factory=lambda: SeasonalNaiveRegressor(),
        params={"type": "seasonal_naive", "lag": 7},
    ),
    "mean": ModelConfig(
        name="mean",
        family="baseline",
        estimator_factory=lambda: MeanRegressor(),
        params={"type": "mean", "window": 30},
    ),
}


def supervised_configs() -> dict[str, ModelConfig]:
    """Configuraciones de modelos supervisados (estrategia global)."""
    from lightgbm import LGBMRegressor

    lgb_params = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    configs = {
        "lightgbm": ModelConfig(
            name="lightgbm",
            family="gbm",
            estimator_factory=lambda: LGBMRegressor(**lgb_params),
            params=dict(lgb_params),
        ),
    }
    return configs


def all_configs() -> dict[str, ModelConfig]:
    """Todas las familias de modelos (baselines + supervisados)."""
    from catboost import CatBoostRegressor
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    from xgboost import XGBRegressor

    configs = dict(BASE_MODELS)
    configs.update(supervised_configs())

    xgb_params = {
        "n_estimators": 300,
        "eta": 0.05,
        "max_depth": 6,
        "random_state": 42,
        "n_jobs": -1,
    }
    configs["xgboost"] = ModelConfig(
        name="xgboost",
        family="gbm",
        estimator_factory=lambda: XGBRegressor(**xgb_params),
        params=dict(xgb_params),
    )

    cb_params = {
        "iterations": 300,
        "learning_rate": 0.05,
        "depth": 6,
        "random_state": 42,
        "verbose": 0,
        "thread_count": -1,
    }
    configs["catboost"] = ModelConfig(
        name="catboost",
        family="gbm",
        estimator_factory=lambda: CatBoostRegressor(**cb_params),
        params=dict(cb_params),
    )

    rf_params = {
        "n_estimators": 200,
        "max_depth": 20,
        "min_samples_leaf": 5,
        "random_state": 42,
        "n_jobs": -1,
    }
    configs["random_forest"] = ModelConfig(
        name="random_forest",
        family="ensemble",
        estimator_factory=lambda: RandomForestRegressor(**rf_params),
        params=dict(rf_params),
    )

    et_params = {
        "n_estimators": 200,
        "max_depth": 20,
        "min_samples_leaf": 5,
        "random_state": 42,
        "n_jobs": -1,
    }
    configs["extra_trees"] = ModelConfig(
        name="extra_trees",
        family="ensemble",
        estimator_factory=lambda: ExtraTreesRegressor(**et_params),
        params=dict(et_params),
    )

    mlp_params = {
        "hidden_layer_sizes": (64, 32),
        "max_iter": 300,
        "random_state": 42,
        "early_stopping": True,
    }
    configs["mlp"] = ModelConfig(
        name="mlp",
        family="neural_net",
        estimator_factory=lambda: MLPRegressor(**mlp_params),
        params=dict(mlp_params),
    )
    return configs


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
        supervised = supervised_configs()
        available = {**BASE_MODELS, **supervised}
        configs = {
            name: available[name]
            for name in args.models.split(",")
            if name in available
        }
    if not configs:
        available = sorted(set(BASE_MODELS) | set(supervised_configs()))
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

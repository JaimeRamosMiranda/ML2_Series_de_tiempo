"""Catálogo central de configuraciones de modelos (Fase 3).

Un único lugar con todas las familias de modelos del proyecto para que la
comparación entre modelos sea siempre con los mismos parámetros. Lo usan
``scripts/train.py`` y el notebook ``04_modelos_adicionales``.

Cada configuración es un :class:`~src.models.train_model.ModelConfig` que
incluye nombre, familia, fábrica de estimador y parámetros (registrados como
params del run de MLflow).
"""

from __future__ import annotations

from src.models.train_model import (
    MeanRegressor,
    ModelConfig,
    NaiveRegressor,
    SeasonalNaiveRegressor,
)


def baseline_configs() -> dict[str, ModelConfig]:
    """Baselines: naive, seasonal-naive y media por serie."""
    return {
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
    """Modelos supervisados con estrategia global (uno para las 150 series).

    Incluye LightGBM, XGBoost, CatBoost, RandomForest, ExtraTrees y MLP.
    """
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    from xgboost import XGBRegressor

    lgb_params = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    xgb_params = {
        "n_estimators": 300,
        "eta": 0.05,
        "max_depth": 6,
        "random_state": 42,
        "n_jobs": -1,
    }
    cb_params = {
        "iterations": 300,
        "learning_rate": 0.05,
        "depth": 6,
        "random_state": 42,
        "verbose": 0,
        "thread_count": -1,
    }
    rf_params = {
        "n_estimators": 200,
        "max_depth": 20,
        "min_samples_leaf": 5,
        "random_state": 42,
        "n_jobs": -1,
    }
    et_params = dict(rf_params)
    mlp_params = {
        "hidden_layer_sizes": (64, 32),
        "max_iter": 300,
        "random_state": 42,
        "early_stopping": True,
    }

    return {
        "lightgbm": ModelConfig(
            name="lightgbm",
            family="gbm",
            estimator_factory=lambda: LGBMRegressor(**lgb_params),
            params=dict(lgb_params),
        ),
        "xgboost": ModelConfig(
            name="xgboost",
            family="gbm",
            estimator_factory=lambda: XGBRegressor(**xgb_params),
            params=dict(xgb_params),
        ),
        "catboost": ModelConfig(
            name="catboost",
            family="gbm",
            estimator_factory=lambda: CatBoostRegressor(**cb_params),
            params=dict(cb_params),
        ),
        "random_forest": ModelConfig(
            name="random_forest",
            family="ensemble",
            estimator_factory=lambda: RandomForestRegressor(**rf_params),
            params=dict(rf_params),
        ),
        "extra_trees": ModelConfig(
            name="extra_trees",
            family="ensemble",
            estimator_factory=lambda: ExtraTreesRegressor(**et_params),
            params=dict(et_params),
        ),
        "mlp": ModelConfig(
            name="mlp",
            family="neural_net",
            estimator_factory=lambda: MLPRegressor(**mlp_params),
            params=dict(mlp_params),
        ),
    }


def all_configs() -> dict[str, ModelConfig]:
    """Todas las familias del proyecto (baselines + supervisados)."""
    return {**baseline_configs(), **supervised_configs()}

#!/usr/bin/env python
"""Generación de la submission con el modelo de producción (Fase 4 + Fase 6).

Carga el modelo productivo (LightGBM) desde la carpeta del repo
``models/demand_forecast`` o desde el Model Registry de MLflow
(``models:/demand_forecast@Production``), reconstruye las features sobre el marco
historial + test y pronostica ene-mar 2018 **día a día** (pronóstico recursivo):
como las features son día-relativas (``lag_1``, ``rolling_*``), predecir los 90
días de una sola vez dejaría NaN desde el día 2; se rellenan las ventas predichas
para que los lags/rolling del día siguiente sean válidos.

Ejemplos:
    python scripts/predict.py                               # modelo del repo
    python scripts/predict.py --model-uri models:/demand_forecast@Production
    python scripts/predict.py --max-series 5                # smoke test
    python scripts/predict.py --output reports/submissions/mi_submission.csv

Requiere los datos crudos (Fase 1): data/raw/train.csv y data/raw/test.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.build_features import FEATURE_COLUMNS, build_features  # noqa: E402

DATA_RAW = ROOT / "data" / "raw"
SUBMISSIONS = ROOT / "reports" / "submissions"
DEFAULT_OUTPUT = SUBMISSIONS / "submission_production.csv"
DEFAULT_MODEL_PATH = ROOT / "models" / "demand_forecast"

DEFAULT_ITEMS = [1, 2, 5, 6, 7, 8, 13, 14, 15, 16, 23, 24, 25, 28, 49]
DEFAULT_STORES = list(range(1, 11))

TRACKING_URI = f"sqlite:///{(ROOT / 'mlruns' / 'mlflow.db').as_posix()}"
TARGET = "sales"


def load_model(model_uri: str):
    """Carga la envoltura pyfunc del modelo productivo.

    ``models:/demand_forecast@Production`` requiere el tracking de MLflow
    configurado (local o DagsHub); una ruta de carpeta se carga directo.
    """
    if str(model_uri).startswith("models:"):
        mlflow.set_tracking_uri(TRACKING_URI)
        return mlflow.pyfunc.load_model(model_uri)
    return mlflow.pyfunc.load_model(str(model_uri))


def build_forecast_frame(
    max_series: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Historial (150 series) + test con ``sales`` en NaN para el periodo futuro."""
    train_raw = pd.read_csv(DATA_RAW / "train.csv", parse_dates=["date"])
    test = pd.read_csv(DATA_RAW / "test.csv", parse_dates=["date"])

    if max_series is not None:
        series_keys = (
            train_raw[["store", "item"]]
            .drop_duplicates()
            .sort_values(["store", "item"])
            .head(max_series)
        )
        pairs = set(zip(series_keys["store"], series_keys["item"]))
        mask = lambda df: df.apply(  # noqa: E731
            lambda r: (r["store"], r["item"]) in pairs, axis=1
        )
    else:
        mask = lambda df: df["store"].isin(DEFAULT_STORES) & df["item"].isin(DEFAULT_ITEMS)  # noqa: E731

    hist = train_raw[mask(train_raw)][["date", "store", "item", "sales"]].copy()
    tst = test[mask(test)][["id", "date", "store", "item"]].copy()

    frame = pd.concat(
        [hist, tst.assign(sales=np.nan)[["date", "store", "item", "sales"]]],
        ignore_index=True,
    ).sort_values(["store", "item", "date"]).reset_index(drop=True)
    return frame, tst


def predict_recursive(model, frame: pd.DataFrame) -> pd.DataFrame:
    """Pronóstico día a día rellenando el marco con las predicciones previas."""
    feat = build_features(frame, target=TARGET)
    features = [c for c in FEATURE_COLUMNS if c in feat.columns]
    forecast_dates = sorted(frame.loc[frame["sales"].isna(), "date"].unique())

    for day, d in enumerate(forecast_dates, 1):
        feat = build_features(frame, target=TARGET)
        idx = feat.index[feat["date"] == d]
        frame.loc[idx, "sales"] = np.maximum(model.predict(feat.loc[idx, features]), 0.0)
        if day % 30 == 0:
            print(f"  día {day:2d}/{len(forecast_dates)} ({d.date()}) predicho")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-uri",
        default=str(DEFAULT_MODEL_PATH),
        help="Carpeta del modelo o URI de MLflow (models:/demand_forecast@Production).",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-series", type=int, default=None,
                        help="Limitar a N series (smoke test rápido).")
    args = parser.parse_args()

    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    model = load_model(args.model_uri)
    print(f"Modelo cargado: {type(model).__name__} desde {args.model_uri}")

    frame, tst = build_forecast_frame(args.max_series)
    print(f"Historial: {len(frame[frame['sales'].notna()]):,} filas | "
          f"A pronosticar: {frame['sales'].isna().sum():,} filas")

    frame = predict_recursive(model, frame)

    pred_df = frame[frame["date"].isin(tst["date"].unique())].merge(
        tst, on=["date", "store", "item"], how="inner"
    )
    assert len(pred_df) == len(tst), "Deben cubrirse todas las filas de test"

    submission = pred_df[["id", "sales"]].sort_values("id").reset_index(drop=True)
    submission["id"] = submission["id"].astype(int)
    submission.to_csv(args.output, index=False)
    print(f"\nGuardada: {args.output}")
    print(f"Filas: {len(submission):,} | Media: {submission['sales'].mean():.1f} unidades/día")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Preprocesamiento del subconjunto de datos (150 series, < 100 MB).

Carga ``data/raw/train.csv`` (913k filas), filtra a las series ``(store, item)``
seleccionadas, divide temporalmente (train <= 2017-09-30; holdout oct-dic 2017)
y construye las features sobre la **serie completa ANTES de dividir**.

Construir las features antes del split permite que el holdout conserve los 90
días completos con lags/variables móviles válidos (se usa solo historial pasado,
sin fuga de datos). El subconjunto por defecto es 10 tiendas × 15 artículos
(150 series), mezcla de demanda alta/media/baja.

Ejemplos:
    python scripts/preprocess.py
    python scripts/preprocess.py --items 1,2,5,6,7,8,13,14,15,16,23,24,25,28,49
    python scripts/preprocess.py --stores 1,2,3

Salidas en ``data/processed/``:
    train.csv, holdout.csv              (subconjunto, sin features)
    train_features.csv, holdout_features.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.build_features import build_features  # noqa: E402

DEFAULT_ITEMS = [1, 2, 5, 6, 7, 8, 13, 14, 15, 16, 23, 24, 25, 28, 49]
DEFAULT_STORES = list(range(1, 11))
TRAIN_CUTOFF = "2017-09-30"
TARGET = "sales"

DATA_RAW = ROOT / "data" / "raw" / "train.csv"
DATA_OUT = ROOT / "data" / "processed"


def load_raw() -> pd.DataFrame:
    """Carga el dataset crudo."""
    df = pd.read_csv(DATA_RAW, parse_dates=["date"])
    return df


def filter_series(
    df: pd.DataFrame,
    stores: list[int] | None = None,
    items: list[int] | None = None,
) -> pd.DataFrame:
    """Filtra el dataset a las series (store, item) seleccionadas."""
    stores = stores or DEFAULT_STORES
    items = items or DEFAULT_ITEMS
    mask = df["store"].isin(stores) & df["item"].isin(items)
    return df[mask].sort_values(["store", "item", "date"]).reset_index(drop=True)


def split_by_date(df: pd.DataFrame, cutoff: str = TRAIN_CUTOFF) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split temporal sin fuga: train <= cutoff, holdout después."""
    cutoff = pd.Timestamp(cutoff)
    train = df[df["date"] <= cutoff].copy()
    holdout = df[df["date"] > cutoff].copy()
    return train, holdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stores", type=str, default=",".join(map(str, DEFAULT_STORES)),
                        help="Tiendas a conservar (por defecto 1-10).")
    parser.add_argument("--items", type=str, default=",".join(map(str, DEFAULT_ITEMS)),
                        help="Artículos a conservar (por defecto los 15 seleccionados).")
    parser.add_argument("--skip-features", action="store_true",
                        help="Solo dividir sin construir features (más rápido).")
    args = parser.parse_args()

    stores = [int(x) for x in args.stores.split(",")]
    items = [int(x) for x in args.items.split(",")]

    print("Cargando data/raw/train.csv ...")
    raw = load_raw()
    full = filter_series(raw, stores=stores, items=items)
    print(f"Filas originales: {len(raw):,} | Subconjunto: {len(full):,} "
          f"({full['store'].nunique()} tiendas x {full['item'].nunique()} items)")

    if args.skip_features:
        train, holdout = split_by_date(full)
        train.to_csv(DATA_OUT / "train.csv", index=False)
        holdout.to_csv(DATA_OUT / "holdout.csv", index=False)
    else:
        print("Construyendo features sobre la serie completa ...")
        featured = build_features(full, target=TARGET)

        train_feat = featured[featured["date"] <= TRAIN_CUTOFF].dropna()
        holdout_feat = featured[featured["date"] > TRAIN_CUTOFF].copy()
        if holdout_feat.isna().any().any():
            print("ADVERTENCIA: hay NaN en el holdout con features; se eliminan.",
                  file=sys.stderr)
            holdout_feat = holdout_feat.dropna()

        train_raw, holdout_raw = split_by_date(full)

        train_raw.to_csv(DATA_OUT / "train.csv", index=False)
        holdout_raw.to_csv(DATA_OUT / "holdout.csv", index=False)
        train_feat.to_csv(DATA_OUT / "train_features.csv", index=False)
        holdout_feat.to_csv(DATA_OUT / "holdout_features.csv", index=False)

        print(f"train.csv            : {len(train_raw):,} filas")
        print(f"holdout.csv          : {len(holdout_raw):,} filas "
              f"({holdout_raw['date'].min().date()} .. {holdout_raw['date'].max().date()})")
        print(f"train_features.csv   : {len(train_feat):,} filas")
        print(f"holdout_features.csv : {len(holdout_feat):,} filas "
              f"({holdout_feat['date'].min().date()} .. {holdout_feat['date'].max().date()})")

    for f in ["train.csv", "holdout.csv", "train_features.csv", "holdout_features.csv"]:
        p = DATA_OUT / f
        if p.exists():
            print(f"  {f:24s} {p.stat().st_size / 1e6:6.2f} MB")


if __name__ == "__main__":
    main()

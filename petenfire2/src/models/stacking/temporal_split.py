"""
temporal_split.py - Split temporal estricto para series de tiempo

Para petenfire2 NUNCA usar split aleatorio. Train 2018-2022, val 2023, test 2024
por defecto, configurable via config.yaml -> data.temporal_split.

Changelog:
- 2026-05-18: Creacion inicial.
"""
from __future__ import annotations

from typing import Tuple

import pandas as pd


def temporal_train_val_test_split(
    df: pd.DataFrame,
    date_col: str,
    config: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide df en train/val/test segun la seccion temporal_split de config.

    Args:
        df: dataframe a dividir
        date_col: nombre de la columna de fecha
        config: dict con keys train_end, val_start, val_end, test_start, test_end
                (acepta tanto config completo como config['data']['temporal_split'])

    Returns:
        (train_df, val_df, test_df)
    """
    # Acepta el config completo o solo la subseccion
    if "data" in config and "temporal_split" in config.get("data", {}):
        ts = config["data"]["temporal_split"]
    elif "temporal_split" in config:
        ts = config["temporal_split"]
    else:
        ts = config

    dates = pd.to_datetime(df[date_col])
    train_end = pd.Timestamp(ts["train_end"])
    val_start = pd.Timestamp(ts["val_start"])
    val_end = pd.Timestamp(ts["val_end"])
    test_start = pd.Timestamp(ts["test_start"])
    test_end = pd.Timestamp(ts["test_end"])

    train = df[dates <= train_end].copy()
    val = df[(dates >= val_start) & (dates <= val_end)].copy()
    test = df[(dates >= test_start) & (dates <= test_end)].copy()

    return train, val, test

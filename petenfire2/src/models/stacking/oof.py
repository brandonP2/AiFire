"""
oof.py - Out-of-fold predictions con folds temporales

Para evitar leakage en series temporales, los folds son anuales:
  fold 0: train 2018,            val 2019
  fold 1: train 2018-2019,       val 2020
  fold 2: train 2018-2020,       val 2021
  fold 3: train 2018-2021,       val 2022
  ...

Cada base learner se entrena n_folds veces y se generan predicciones OOF
para el set completo de entrenamiento.

Changelog:
- 2026-05-18: Creacion inicial.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Optional

import numpy as np
import pandas as pd


def generate_temporal_oof(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    learner,
    n_folds: int = 5,
    min_train_years: int = 1,
) -> np.ndarray:
    """Genera OOF predictions con folds temporales anuales.

    Args:
        X: features del training set
        y: target del training set
        dates: serie de fechas (datetime) alineada con X
        learner: instancia de BaseLearner (se hace deepcopy por fold)
        n_folds: cantidad maxima de folds (uno por año posterior al min)
        min_train_years: cantidad minima de años en train para el primer fold

    Returns:
        array de probabilidades OOF [n_samples]. Las filas que no caen
        en ningun fold de validacion quedan en NaN.
    """
    dates = pd.to_datetime(dates).reset_index(drop=True)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    years = sorted(dates.dt.year.unique())
    if len(years) < min_train_years + 1:
        raise ValueError(
            f"Se necesitan al menos {min_train_years + 1} años para OOF temporal"
        )

    oof = np.full(len(X), np.nan, dtype=float)

    val_years = years[min_train_years:]
    if n_folds is not None:
        val_years = val_years[:n_folds]

    for fold_idx, val_year in enumerate(val_years):
        train_mask = dates.dt.year < val_year
        val_mask = dates.dt.year == val_year

        if train_mask.sum() == 0 or val_mask.sum() == 0:
            continue

        fold_learner = deepcopy(learner)
        fold_learner.fit(
            X.loc[train_mask], y.loc[train_mask],
            X.loc[val_mask], y.loc[val_mask],
        )
        proba = fold_learner.predict_proba(X.loc[val_mask])[:, 1]
        oof[val_mask.values] = proba

    return oof

"""
ensemble.py - StackingEnsemble orquestador

Coordina:
  1. OOF predictions de cada base learner (folds temporales)
  2. Reentrenamiento de cada base learner en todo el training set
  3. Entrenamiento del meta-learner con las OOF
  4. Prediccion: base_learners en X -> meta_features -> meta_learner

Uso:
    ensemble = StackingEnsemble(
        base_learners=[LGBMBaseLearner(), XGBBaseLearner(), CatBoostBaseLearner(), RFBaseLearner()],
        meta_learner=MetaLearner(),
    )
    ensemble.fit(X_train, y_train, dates_train, X_val=X_val, y_val=y_val)
    proba = ensemble.predict_proba(X_test)

Changelog:
- 2026-05-18: Creacion inicial.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from .base_learners import BaseLearner
from .meta_learner import MetaLearner
from .oof import generate_temporal_oof


class StackingEnsemble:
    """Stacking de base learners + meta-learner calibrado."""

    def __init__(
        self,
        base_learners: List[BaseLearner],
        meta_learner: Optional[MetaLearner] = None,
        n_folds: int = 5,
    ):
        self.base_learners = base_learners
        self.meta_learner = meta_learner or MetaLearner()
        self.n_folds = n_folds
        self.oof_predictions_: Optional[np.ndarray] = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dates: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ):
        # 1) OOF predictions para cada base learner
        oof = np.zeros((len(X), len(self.base_learners)), dtype=float)
        for i, learner in enumerate(self.base_learners):
            oof[:, i] = generate_temporal_oof(X, y, dates, learner, n_folds=self.n_folds)

        # 2) Reentrenar cada base learner en todo el training set (con val opcional)
        for learner in self.base_learners:
            learner.fit(X, y, X_val, y_val)

        # 3) Entrenar meta-learner sobre OOF (solo filas no-NaN)
        valid_mask = ~np.isnan(oof).any(axis=1)
        self.oof_predictions_ = oof
        self.meta_learner.fit(oof[valid_mask], y.values[valid_mask])

        return self

    def _build_meta_features(self, X: pd.DataFrame) -> np.ndarray:
        meta = np.zeros((len(X), len(self.base_learners)), dtype=float)
        for i, learner in enumerate(self.base_learners):
            meta[:, i] = learner.predict_proba(X)[:, 1]
        return meta

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        meta = self._build_meta_features(X)
        return self.meta_learner.predict_proba(meta)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)

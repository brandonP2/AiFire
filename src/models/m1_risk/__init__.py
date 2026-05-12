from __future__ import annotations

import numpy as np


class CalibratedModel:
    """Wrapper que aplica calibración de probabilidades sobre un modelo base.

    Definido aquí (no en train.py) para que joblib lo serialice con un módulo
    estable (src.models.m1_risk) en vez de __main__.
    """

    def __init__(self, base_model, calibrator) -> None:
        self._base = base_model
        self._cal = calibrator

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self._base.predict_proba(X)[:, 1]
        cal = np.clip(self._cal.predict(raw), 0.0, 1.0)
        return np.column_stack([1.0 - cal, cal])

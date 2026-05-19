"""
meta_learner.py - Meta-learner para el stacking ensemble

Combina las predicciones OOF de los 4 base learners con LogisticRegression
y calibra la salida con regresion isotonica.

Changelog:
- 2026-05-18: Creacion inicial.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression


class MetaLearner:
    """LogisticRegression + calibracion isotonica."""

    def __init__(
        self,
        C: float = 1.0,
        class_weight: Optional[str] = "balanced",
        random_state: int = 42,
        calibration_cv: int = 3,
    ):
        self.C = C
        self.class_weight = class_weight
        self.random_state = random_state
        self.calibration_cv = calibration_cv
        self.model = None

    def fit(self, X_meta: np.ndarray, y: np.ndarray):
        """Entrena LogisticRegression y la envuelve con calibracion isotonica."""
        base = LogisticRegression(
            C=self.C,
            class_weight=self.class_weight,
            random_state=self.random_state,
            max_iter=1000,
        )
        self.model = CalibratedClassifierCV(
            estimator=base, method="isotonic", cv=self.calibration_cv,
        )
        self.model.fit(X_meta, y)
        return self

    def predict_proba(self, X_meta: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X_meta)

    def predict(self, X_meta: np.ndarray) -> np.ndarray:
        return self.model.predict(X_meta)

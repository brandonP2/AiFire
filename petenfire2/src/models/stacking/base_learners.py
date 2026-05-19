"""
base_learners.py - Base learners para el stacking ensemble de petenfire2

Define 4 wrappers homogeneos sobre LightGBM, XGBoost, CatBoost y RandomForest.
Todos los wrappers exponen:
  - fit(X_train, y_train, X_val=None, y_val=None) con early stopping cuando aplica
  - predict_proba(X) -> np.ndarray [n, 2]
  - get_feature_importance() -> dict[str, float]

Changelog:
- 2026-05-18: Creacion inicial.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

try:
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation
except ImportError:
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None


def _compute_scale_pos_weight(y: pd.Series) -> float:
    """Razon negativos/positivos para balancear clases."""
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    if pos == 0:
        return 1.0
    return neg / pos


@dataclass
class BaseLearner:
    """Wrapper base. Subclases setean self.model en fit()."""
    name: str = "base"
    params: Dict[str, Any] = field(default_factory=dict)
    random_state: int = 42
    model: Any = None
    feature_names_: Optional[list] = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        raise NotImplementedError

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def get_feature_importance(self) -> Dict[str, float]:
        if self.model is None:
            return {}
        importance = getattr(self.model, "feature_importances_", None)
        if importance is None:
            return {}
        names = self.feature_names_ or [f"f{i}" for i in range(len(importance))]
        return dict(zip(names, [float(v) for v in importance]))


class LGBMBaseLearner(BaseLearner):
    """LightGBM con scale_pos_weight y metric average_precision."""

    def __init__(self, params: Optional[Dict[str, Any]] = None, random_state: int = 42):
        if LGBMClassifier is None:
            raise ImportError("lightgbm no esta instalado")
        default = {
            "n_estimators": 2000,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "max_depth": -1,
            "min_child_samples": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "binary",
            "metric": "average_precision",
            "verbose": -1,
        }
        default.update(params or {})
        super().__init__(name="lightgbm", params=default, random_state=random_state)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        spw = _compute_scale_pos_weight(y_train)
        params = dict(self.params)
        params.setdefault("scale_pos_weight", spw)
        params.setdefault("random_state", self.random_state)

        self.model = LGBMClassifier(**params)
        self.feature_names_ = list(getattr(X_train, "columns", []))

        eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None
        callbacks = []
        if eval_set:
            callbacks = [early_stopping(stopping_rounds=50, verbose=False), log_evaluation(0)]

        self.model.fit(X_train, y_train, eval_set=eval_set, callbacks=callbacks)
        return self


class XGBBaseLearner(BaseLearner):
    """XGBoost con scale_pos_weight y eval_metric aucpr."""

    def __init__(self, params: Optional[Dict[str, Any]] = None, random_state: int = 42):
        default = {
            "n_estimators": 2000,
            "learning_rate": 0.05,
            "max_depth": 6,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "tree_method": "hist",
            "early_stopping_rounds": 50,
        }
        default.update(params or {})
        super().__init__(name="xgboost", params=default, random_state=random_state)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        spw = _compute_scale_pos_weight(y_train)
        params = dict(self.params)
        params.setdefault("scale_pos_weight", spw)
        params.setdefault("random_state", self.random_state)

        # Si no hay validacion, removemos early_stopping_rounds
        if X_val is None or y_val is None:
            params.pop("early_stopping_rounds", None)

        self.model = XGBClassifier(**params)
        self.feature_names_ = list(getattr(X_train, "columns", []))

        eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None
        if eval_set is not None:
            self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        else:
            self.model.fit(X_train, y_train, verbose=False)
        return self


class CatBoostBaseLearner(BaseLearner):
    """CatBoost con auto_class_weights Balanced y eval_metric PRAUC."""

    def __init__(self, params: Optional[Dict[str, Any]] = None, random_state: int = 42):
        if CatBoostClassifier is None:
            raise ImportError("catboost no esta instalado")
        default = {
            "iterations": 2000,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 3,
            "auto_class_weights": "Balanced",
            "eval_metric": "PRAUC",
            "od_type": "Iter",
            "od_wait": 50,
            "verbose": 0,
        }
        default.update(params or {})
        super().__init__(name="catboost", params=default, random_state=random_state)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        params = dict(self.params)
        params.setdefault("random_seed", self.random_state)

        self.model = CatBoostClassifier(**params)
        self.feature_names_ = list(getattr(X_train, "columns", []))

        eval_set = (X_val, y_val) if X_val is not None and y_val is not None else None
        self.model.fit(X_train, y_train, eval_set=eval_set, use_best_model=eval_set is not None)
        return self


class RFBaseLearner(BaseLearner):
    """RandomForest con class_weight balanced. No tiene early stopping nativo."""

    def __init__(self, params: Optional[Dict[str, Any]] = None, random_state: int = 42):
        default = {
            "n_estimators": 500,
            "max_depth": 20,
            "min_samples_split": 50,
            "min_samples_leaf": 20,
            "class_weight": "balanced",
            "n_jobs": -1,
        }
        default.update(params or {})
        super().__init__(name="random_forest", params=default, random_state=random_state)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        params = dict(self.params)
        params.setdefault("random_state", self.random_state)

        self.model = RandomForestClassifier(**params)
        self.feature_names_ = list(getattr(X_train, "columns", []))
        self.model.fit(X_train, y_train)
        return self

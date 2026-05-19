"""
analytics.py - Modelado y evaluacion para petenfire2

Adaptado de la plantilla model-template-build-uv:
- Sin SMOTE (datasets de 136M filas no caben en RAM)
- Soporte para LightGBM, XGBoost, CatBoost y RandomForest
- Split temporal en lugar de aleatorio
- Funciones reutilizables: threshold_iter, choose_best, evaluate_shap_values

Changelog:
- 2026-05-18: Creacion inicial adaptando la plantilla a petenfire2.
"""
import inspect
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import shap
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from xgboost import XGBClassifier

import utils
import visualization

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None


# =============================================================================
# Splits
# =============================================================================

def df_split(df: pd.DataFrame, objective: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a dataframe into variables and objective."""
    df_x = df.drop(objective, axis=1)
    df_y = df[objective]
    return df_x, df_y


def temporal_split(
    df: pd.DataFrame,
    date_column: str,
    train_end: str,
    val_start: str,
    val_end: str,
    test_start: str,
    test_end: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split temporal estricto. No shuffle, no leakage.

    Args:
        df: dataframe con la columna de fecha
        date_column: nombre de la columna de fecha
        train_end: ultima fecha incluida en train (YYYY-MM-DD)
        val_start: primera fecha de validacion
        val_end: ultima fecha de validacion
        test_start: primera fecha de test
        test_end: ultima fecha de test
    """
    dates = pd.to_datetime(df[date_column])
    train = df[dates <= pd.Timestamp(train_end)]
    val = df[(dates >= pd.Timestamp(val_start)) & (dates <= pd.Timestamp(val_end))]
    test = df[(dates >= pd.Timestamp(test_start)) & (dates <= pd.Timestamp(test_end))]
    return train, val, test


# =============================================================================
# Threshold utilities (reutilizables de la plantilla)
# =============================================================================

def threshold_truncate(
    real_values: pd.Series, predicted_probabilities: np.array, threshold: float
) -> dict:
    """Calcula metricas para un threshold dado."""
    prediction = (predicted_probabilities >= threshold).astype(int)
    precision, recall, f1, o = precision_recall_fscore_support(
        real_values, prediction, average="binary", labels=[1, 0]
    )
    tn, fp, fn, tp = confusion_matrix(real_values, prediction).ravel()
    return {
        "threshold": threshold,
        "positive_prediction": prediction.mean(),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": o,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def threshold_iter(
    real_values: pd.Series,
    predicted_probabilities: np.array,
    min_value: float = 0,
    max_value: float = 1,
    step: float = 0.1,
) -> List[dict]:
    """Itera sobre thresholds calculando metricas."""
    if min_value < 0:
        min_value = 0.0
    if max_value > 1:
        max_value = 1.0

    logger = utils.get_logger()
    results = []
    for threshold in np.arange(min_value, max_value, step):
        logger.debug("On threshold %s", threshold)
        results.append(threshold_truncate(real_values, predicted_probabilities, threshold))
    return results


def choose_best(
    metrics: List[dict],
    measure: Union[str, Callable] = "precision",
    search: str = "max",
    constraint: Callable = None,
) -> dict:
    """Selecciona el mejor threshold segun un criterio.

    Args:
        metrics: lista de metricas por threshold
        measure: 'precision', 'recall', 'f1' o callable
        search: 'max' o 'min'
        constraint: callable(metric_dict) -> bool. Filtra antes de elegir.
    """
    if callable(measure):
        function = measure
    else:
        if measure not in ("precision", "recall", "f1"):
            measure = "precision"
        function = lambda x: x[measure]

    candidates = metrics
    if constraint is not None:
        filtered = [m for m in metrics if constraint(m)]
        if filtered:
            candidates = filtered

    if search == "max":
        return max(candidates, key=function)
    return min(candidates, key=function)


def choose_threshold_by_strategy(
    metrics: List[dict],
    strategy: str = "max_recall",
    min_precision: float = 0.2,
    min_recall: float = 0.25,
) -> dict:
    """Estrategias de threshold segun config.yaml.

    Estrategias:
      - max_precision: maximiza precision con recall >= min_recall
      - max_f1: maximiza f1 con recall >= min_recall
      - max_recall: maximiza recall con precision >= min_precision
      - balance: precision aproximadamente igual a recall
    """
    if strategy == "max_precision":
        return choose_best(
            metrics, measure="precision", search="max",
            constraint=lambda m: m["recall"] >= min_recall,
        )
    if strategy == "max_f1":
        return choose_best(
            metrics, measure="f1", search="max",
            constraint=lambda m: m["recall"] >= min_recall,
        )
    if strategy == "max_recall":
        return choose_best(
            metrics, measure="recall", search="max",
            constraint=lambda m: m["precision"] >= min_precision,
        )
    if strategy == "balance":
        return choose_best(
            metrics, measure=lambda m: -abs(m["precision"] - m["recall"]), search="max",
        )
    raise ValueError(f"Unknown threshold strategy: {strategy}")


# =============================================================================
# Sampling - solo undersampling (SMOTE no soportado en petenfire2)
# =============================================================================

def under_sampling(
    variables: pd.DataFrame, objective: pd.Series, ratio: float, random_state: int = 123
) -> Tuple[pd.DataFrame, pd.Series]:
    """Random undersampling de la clase mayoritaria."""
    under = RandomUnderSampler(random_state=random_state, sampling_strategy=ratio)
    return under.fit_resample(variables, objective)


def balance_data(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    under_sampling_ratio: float = None,
    random_state: int = 123,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Balancea con undersampling unicamente. SMOTE deshabilitado en petenfire2."""
    logger = utils.get_logger()
    if under_sampling_ratio:
        logger.debug("Undersample by %s", under_sampling_ratio)
        train_x, train_y = under_sampling(
            train_x, train_y, under_sampling_ratio, random_state=random_state
        )
    return train_x, train_y


# =============================================================================
# SHAP (reutilizable de la plantilla)
# =============================================================================

def evaluate_shap_values(
    model,
    test: pd.DataFrame,
    plot: bool = True,
    max_display: int = None,
) -> Tuple[np.array, np.array, shap.TreeExplainer]:
    """Calcula y opcionalmente plotea SHAP values para modelos tree-based."""
    explainer_sample_size = min(100, len(test))
    if explainer_sample_size == len(test):
        explainer_data = test.copy()
    else:
        explainer_data = test.sample(n=explainer_sample_size)

    explainer = shap.TreeExplainer(
        model,
        feature_perturbation="interventional",
        model_output="probability",
        data=explainer_data,
    )
    shap_values = explainer(test)
    predictions = model.predict_proba(test)[:, 1]
    if plot:
        visualization.plot_shap_values(shap_values, max_display=max_display)
    return shap_values, predictions, explainer


def get_dataframe_filter_features(df: pd.DataFrame, feature_list: List[str]) -> pd.DataFrame:
    """Subset de columnas existentes."""
    return df[df.columns.intersection(feature_list)]


# =============================================================================
# Entrenamiento de modelos
# =============================================================================

def _build_model(model_type: str, hyperparameters: dict, class_weight: str = None):
    """Construye un modelo segun el tipo solicitado."""
    hp = dict(hyperparameters or {})

    if model_type == "logistic_regression":
        return LogisticRegression(**hp)
    if model_type == "logistic_regression_balanced":
        return LogisticRegression(class_weight="balanced", **hp)
    if model_type == "xgboost":
        return XGBClassifier(**hp)
    if model_type == "adaboost":
        return AdaBoostClassifier(**hp)
    if model_type == "random_forest":
        if class_weight and "class_weight" not in hp:
            hp["class_weight"] = class_weight
        return RandomForestClassifier(**hp)
    if model_type == "lightgbm":
        if LGBMClassifier is None:
            raise ImportError("lightgbm no esta instalado")
        if class_weight and "class_weight" not in hp:
            hp["class_weight"] = class_weight
        return LGBMClassifier(**hp)
    if model_type == "catboost":
        if CatBoostClassifier is None:
            raise ImportError("catboost no esta instalado")
        if class_weight == "balanced" and "auto_class_weights" not in hp:
            hp["auto_class_weights"] = "Balanced"
        return CatBoostClassifier(verbose=0, **hp)
    raise ValueError(f"Unknown model_type: {model_type}")


def train_model_from_df(
    df: pd.DataFrame,
    objective: str,
    scaler_type: str,
    under_sampling_ratio: float,
    model_type: str,
    random_state: int = 123,
    hyperparameters: dict = None,
    class_weight: str = None,
) -> Tuple[object, pd.DataFrame, pd.Series]:
    """Entrena un modelo desde un dataframe.

    No usa SMOTE. Soporta logistic_regression, xgboost, adaboost,
    random_forest, lightgbm, catboost.
    """
    logger = utils.get_logger()

    logger.debug("Split data into variables and objective")
    train_x, train_y = df_split(df, objective)

    if scaler_type:
        logger.debug("Use scaler: %s", scaler_type)
        scaler = StandardScaler()
        if scaler_type == "Robust":
            scaler = RobustScaler()
        elif scaler_type == "MinMax":
            scaler = MinMaxScaler()
        scaler.fit(train_x)
        train_x = pd.DataFrame(scaler.transform(train_x), columns=train_x.columns)

    train_x, train_y = balance_data(
        train_x, train_y, under_sampling_ratio, random_state=random_state
    )

    hp = dict(hyperparameters or {})
    # CatBoost no acepta random_state directamente (usa random_seed)
    if model_type == "catboost":
        hp.setdefault("random_seed", random_state)
    else:
        hp.setdefault("random_state", random_state)

    model = _build_model(model_type, hp, class_weight=class_weight)
    model.fit(train_x, train_y)
    return model, train_x, train_y


def train_with_temporal_validation(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    objective_column: str,
    model_type: str,
    hyperparameters: dict = None,
    random_state: int = 123,
    scaler_type: str = None,
    under_sampling_ratio: float = None,
    class_weight: str = None,
    threshold_strategy: str = "max_recall",
    min_precision: float = 0.2,
    min_recall: float = 0.25,
    threshold_step: float = 0.05,
    plot_threshold: bool = False,
) -> dict:
    """Entrena con train set y valida temporalmente con val set."""
    logger = utils.get_logger()

    val_x, val_y = df_split(df_val, objective_column)

    model, _, _ = train_model_from_df(
        df_train,
        objective_column,
        scaler_type,
        under_sampling_ratio,
        model_type,
        random_state=random_state,
        hyperparameters=hyperparameters,
        class_weight=class_weight,
    )

    val_proba = model.predict_proba(val_x)[:, 1]
    iters = threshold_iter(val_y, val_proba, min_value=threshold_step, max_value=1, step=threshold_step)

    if plot_threshold:
        visualization.plot_threshold_metrics(pd.DataFrame(iters))

    best = choose_threshold_by_strategy(
        iters, strategy=threshold_strategy,
        min_precision=min_precision, min_recall=min_recall,
    )
    fpr, tpr, _ = roc_curve(val_y, val_proba)
    best["val_roc_auc"] = auc(fpr, tpr)
    best["val_pr_auc"] = average_precision_score(val_y, val_proba)
    best["model"] = model

    logger.info(
        "Val | P: %.4f R: %.4f F1: %.4f ROC: %.4f PR: %.4f",
        best["precision"], best["recall"], best["f1"],
        best["val_roc_auc"], best["val_pr_auc"],
    )
    return best


# =============================================================================
# Feature utilities (reutilizables)
# =============================================================================

def get_features(
    df_importance: pd.DataFrame, features_amount: int, fixed_features: List[str] = None
) -> List[str]:
    """Selecciona features respetando una lista de fixed_features."""
    features = list(df_importance.feature_name[0:features_amount])

    if fixed_features is None or len(fixed_features) == 0:
        return features

    extra_features = len(set(features) | set(fixed_features)) - features_amount
    if extra_features <= 0:
        return features

    to_remove = []
    for x in reversed(features):
        if x not in fixed_features:
            to_remove.append(x)
        if len(to_remove) >= extra_features:
            break

    cleaned = [x for x in features if x not in to_remove]
    selected = list(set(cleaned) | set(fixed_features))
    selected.sort()
    return selected


def transformed_features_to_raw(transformed_features: List[str], features_map: dict) -> List[str]:
    """Mapeo inverso del GenericTransformer."""
    reverse = {}
    for old, news in features_map.items():
        for new in news:
            reverse[new] = old
    raw = []
    for tf in transformed_features:
        rf = reverse.get(tf, tf)
        if rf not in raw:
            raw.append(rf)
    return raw


def raw_features_to_transformed(raw_features: List[str], features_map: dict) -> List[str]:
    """Mapeo directo del GenericTransformer."""
    transformed = []
    for rf in raw_features:
        news = features_map.get(rf, [rf])
        for nf in news:
            if nf not in transformed:
                transformed.append(nf)
    return transformed


def load_model_from_local(model_path: str):
    """Carga modelo desde joblib."""
    logger = utils.get_logger()
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    logger.info("Loading model from: %s", model_path)
    return joblib.load(model_path)


def transform_to_basic_type(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte tipos nullables de pandas a tipos basicos."""
    map_types = {"Int32": "int32", "Int64": "int64", "Float32": "float32", "Float64": "float64"}
    for old, new in map_types.items():
        for col in df.select_dtypes(include=old).columns:
            df[col] = df[col].astype(new)
    return df

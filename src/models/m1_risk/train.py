"""Entrenamiento del modelo M1 de riesgo de incendio forestal.

Pipeline:
1. Carga el dataset M1 (data/processed/m1_dataset.parquet).
2. Aplica SMOTE conservador sobre el conjunto train para mitigar el desbalance (~1-2%).
3. Entrena XGBoost + LightGBM con validación temporal estricta.
4. Calibra probabilidades con Platt scaling (CalibratedClassifierCV isotónico).
5. Evalúa sobre val y test con métricas apropiadas para clases desbalanceadas.
6. Guarda el mejor modelo (por AUC-PR sobre val) en models_artifacts/m1/.

Validación temporal estricta:
    train: 2018-2022 | val: 2023 | test: 2024
    Nunca K-fold aleatorio sobre series temporales.

Uso:
    uv run python -m src.models.m1_risk.train
    uv run python -m src.models.m1_risk.train --model xgboost --n-estimators 500
    uv run python -m src.models.m1_risk.train --no-smote
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer
import yaml
from loguru import logger
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data.config import Settings
from src.data.config import settings as default_settings
from src.models.m1_risk import CalibratedModel

app = typer.Typer(add_completion=False)

# ---------------------------------------------------------------------------
# Features del modelo M1
# ---------------------------------------------------------------------------

FEATURE_COLS: list[str] = [
    # Clima
    "T2M", "RH2M", "WS10M", "PRECTOTCORR",
    # FWI
    "fwi", "ffmc_val", "dmc_val", "dc_val", "isi_val", "bui_val",
    # Precipitación acumulada
    "prec_acc7d", "prec_acc14d",
    # Vegetación
    "ndvi", "ndvi_lag7", "ndvi_lag14",
    # Topografía
    "elevation_m", "slope_deg", "aspect_deg",
    # Antrópico
    "dist_roads_km", "dist_settlements_km", "is_protected_area",
    # Temporal (cíclico)
    "month", "day_of_year",
]

TARGET_COL = "fire_occurred"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_splits(dataset_path: Path) -> tuple[
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
]:
    """Carga el dataset y separa train/val/test según la columna split.

    Args:
        dataset_path: Ruta al parquet M1.

    Returns:
        Tupla (X_train, y_train, X_val, y_val, X_test, y_test).

    Raises:
        FileNotFoundError: Si el dataset no existe.
        ValueError: Si faltan columnas requeridas o no hay datos suficientes.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset no encontrado: {dataset_path}. "
            "Ejecuta primero: uv run python -m src.features.build_dataset"
        )

    logger.info(f"Cargando dataset desde {dataset_path}...")
    df = pd.read_parquet(dataset_path)
    logger.info(f"Dataset cargado: {len(df):,} filas x {df.shape[1]} columnas")

    # Validar columnas
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        logger.warning(f"Columnas de features faltantes: {missing}. Se usarán las disponibles.")
    if not available_features:
        raise ValueError("No hay features disponibles en el dataset.")
    if TARGET_COL not in df.columns:
        raise ValueError(f"Columna target '{TARGET_COL}' no encontrada.")
    if "split" not in df.columns:
        raise ValueError("Columna 'split' no encontrada.")

    # Separar splits
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        n = len(split_df)
        n_fire = split_df[TARGET_COL].sum()
        logger.info(f"  {name}: {n:,} filas, {n_fire:,} fuego ({100*n_fire/n:.2f}%)")

    X_train = train_df[available_features].values.astype(np.float32)
    y_train = train_df[TARGET_COL].values.astype(np.int32)
    X_val = val_df[available_features].values.astype(np.float32)
    y_val = val_df[TARGET_COL].values.astype(np.int32)
    X_test = test_df[available_features].values.astype(np.float32)
    y_test = test_df[TARGET_COL].values.astype(np.int32)

    return X_train, y_train, X_val, y_val, X_test, y_test


def _apply_smote(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
    sampling_strategy: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Aplica SMOTE conservador para mitigar el desbalance extremo de clases.

    Args:
        X: Features de entrenamiento.
        y: Etiquetas de entrenamiento.
        random_state: Semilla.
        sampling_strategy: Ratio final minoritaria/mayoritaria (default 10%).

    Returns:
        (X_resampled, y_resampled)
    """
    from imblearn.over_sampling import SMOTE

    from sklearn.impute import SimpleImputer

    fire_rate = y.mean()
    logger.info(f"Aplicando SMOTE: tasa de fuego actual = {100*fire_rate:.2f}%")

    # SMOTE no acepta NaN — imputar medianas antes del resampleo
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        random_state=random_state,
        k_neighbors=5,
    )
    X_res, y_res = smote.fit_resample(X_imp, y)
    new_fire_rate = y_res.mean()
    logger.info(
        f"SMOTE aplicado: {len(X):,} → {len(X_res):,} muestras, "
        f"tasa de fuego = {100*new_fire_rate:.2f}%"
    )
    return X_res, y_res


def _compute_scale_pos_weight(y: np.ndarray) -> float:
    """Calcula scale_pos_weight para XGBoost/LightGBM desde el ratio de clases."""
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    if n_pos == 0:
        raise ValueError("No hay muestras positivas (fire_occurred=True) en train.")
    return float(n_neg) / float(n_pos)


def _evaluate(
    model,
    X: np.ndarray,
    y: np.ndarray,
    split_name: str,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Evalúa el modelo sobre un split y loggea las métricas.

    Args:
        model: Modelo ajustado con método predict_proba.
        X: Features del split.
        y: Etiquetas del split.
        split_name: Nombre del split para logging.
        threshold: Umbral de clasificación.

    Returns:
        Diccionario de métricas.
    """
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)

    auc_roc = roc_auc_score(y, proba)
    auc_pr = average_precision_score(y, proba)
    precision = precision_score(y, pred, zero_division=0)
    recall = recall_score(y, pred, zero_division=0)
    f1 = f1_score(y, pred, zero_division=0)

    metrics = {
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    logger.info(
        f"{split_name:6s} | AUC-ROC={auc_roc:.4f} | AUC-PR={auc_pr:.4f} | "
        f"P={precision:.4f} | R={recall:.4f} | F1={f1:.4f}"
    )
    return metrics


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_estimators: int = 500,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    random_state: int = 42,
) -> object:
    """Entrena un XGBoostClassifier con early stopping sobre val.

    Args:
        X_train: Features de entrenamiento.
        y_train: Etiquetas de entrenamiento.
        X_val: Features de validación (para early stopping).
        y_val: Etiquetas de validación.
        n_estimators: Número máximo de árboles.
        max_depth: Profundidad máxima de cada árbol.
        learning_rate: Tasa de aprendizaje.
        random_state: Semilla.

    Returns:
        XGBClassifier ajustado.
    """
    from xgboost import XGBClassifier

    scale_pos_weight = _compute_scale_pos_weight(y_train)
    logger.info(f"XGBoost: scale_pos_weight={scale_pos_weight:.1f}")

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        eval_metric="aucpr",
        early_stopping_rounds=50,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )
    logger.info(f"XGBoost: mejor iteración = {model.best_iteration}")
    return model


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    n_estimators: int = 500,
    max_depth: int = -1,
    learning_rate: float = 0.05,
    random_state: int = 42,
) -> object:
    """Entrena un LGBMClassifier con early stopping sobre val.

    Args:
        X_train: Features de entrenamiento.
        y_train: Etiquetas de entrenamiento.
        X_val: Features de validación.
        y_val: Etiquetas de validación.
        feature_names: Nombres de las columnas de features.
        n_estimators: Número máximo de árboles.
        max_depth: Profundidad máxima (-1 = sin límite).
        learning_rate: Tasa de aprendizaje.
        random_state: Semilla.

    Returns:
        LGBMClassifier ajustado.
    """
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation

    scale_pos_weight = 1.0  # SMOTE ya balanceó las clases
    logger.info(f"LightGBM: scale_pos_weight={scale_pos_weight:.1f}")

    model = LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="average_precision",
        feature_name=feature_names,
        callbacks=[early_stopping(30, verbose=False), log_evaluation(50)],
    )
    logger.info(f"LightGBM: mejor iteración = {model.best_iteration_}")
    return model


def calibrate_model(
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    method: str = "isotonic",
) -> CalibratedModel:
    """Calibra las probabilidades del modelo usando regresión isotónica o Platt.

    Args:
        model: Modelo base ya entrenado.
        X_val: Features del conjunto de calibración.
        y_val: Etiquetas del conjunto de calibración.
        method: 'isotonic' o 'sigmoid' (Platt scaling).

    Returns:
        _CalibratedModel listo para predict_proba.
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    logger.info(f"Calibrando probabilidades ({method})...")
    raw_proba = model.predict_proba(X_val)[:, 1]

    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw_proba, y_val)
    else:
        calibrator = LogisticRegression()
        calibrator.fit(raw_proba.reshape(-1, 1), y_val)

    return CalibratedModel(model, calibrator)


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------


def train_m1(
    cfg: Settings | None = None,
    dataset_path: Path | None = None,
    output_dir: Path | None = None,
    model_type: str = "lightgbm",
    n_estimators: int = 500,
    use_smote: bool = True,
    smote_strategy: float = 0.1,
    calibration_method: str = "isotonic",
    threshold: float = 0.05,
) -> dict[str, dict[str, float]]:
    """Entrena y evalúa el modelo M1.

    Args:
        cfg: Configuración del proyecto.
        dataset_path: Ruta al parquet M1.
        output_dir: Directorio donde guardar el modelo y métricas.
        model_type: 'xgboost' o 'lightgbm'.
        n_estimators: Número máximo de árboles.
        use_smote: Si aplicar SMOTE antes del entrenamiento.
        smote_strategy: Ratio SMOTE (minoritaria/mayoritaria tras resampleo).
        calibration_method: 'sigmoid' o 'isotonic'.
        threshold: Umbral de clasificación para métricas discretas.

    Returns:
        Diccionario de métricas por split {'val': {...}, 'test': {...}}.
    """
    import joblib

    cfg = cfg or default_settings
    dataset_path = dataset_path or (cfg.data_processed / "m1_dataset.parquet")
    output_dir = output_dir or (cfg.project_root / "models_artifacts" / "m1")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Cargar datos
    X_train, y_train, X_val, y_val, X_test, y_test = _load_splits(dataset_path)

    # Features disponibles (columnas del parquet que coinciden con FEATURE_COLS)
    import pyarrow.parquet as pq
    dataset_cols = set(pq.read_schema(str(dataset_path)).names)
    available_features = [c for c in FEATURE_COLS if c in dataset_cols]

    # Descartar features completamente NaN (ej. NDVI sin datos descargados)
    all_nan_mask = np.isnan(X_train).all(axis=0)
    if all_nan_mask.any():
        dropped = [f for f, drop in zip(available_features, all_nan_mask) if drop]
        logger.warning(f"Descartando {len(dropped)} features completamente NaN: {dropped}")
        available_features = [f for f, drop in zip(available_features, all_nan_mask) if not drop]
        X_train = X_train[:, ~all_nan_mask]
        X_val = X_val[:, ~all_nan_mask]
        X_test = X_test[:, ~all_nan_mask]

    # SMOTE
    if use_smote:
        X_train, y_train = _apply_smote(
            X_train, y_train,
            random_state=cfg.random_seed,
            sampling_strategy=smote_strategy,
        )

    # Entrenamiento
    if model_type.lower() == "xgboost":
        base_model = train_xgboost(
            X_train, y_train, X_val, y_val,
            n_estimators=n_estimators,
            random_state=cfg.random_seed,
        )
    elif model_type.lower() == "lightgbm":
        base_model = train_lightgbm(
            X_train, y_train, X_val, y_val,
            feature_names=available_features,
            n_estimators=n_estimators,
            random_state=cfg.random_seed,
        )
    else:
        raise ValueError(f"model_type desconocido: {model_type}. Usar 'xgboost' o 'lightgbm'.")

    # Calibración
    calibrated_model = calibrate_model(base_model, X_val, y_val, method=calibration_method)

    # Evaluación
    logger.info("--- Evaluación ---")
    val_metrics = _evaluate(calibrated_model, X_val, y_val, "val", threshold)
    test_metrics = _evaluate(calibrated_model, X_test, y_test, "test", threshold)

    all_metrics = {"val": val_metrics, "test": test_metrics}

    # Guardar modelo
    model_path = output_dir / f"m1_{model_type}_calibrated.joblib"
    joblib.dump(calibrated_model, model_path)
    logger.success(f"Modelo guardado en {model_path}")

    # Guardar métricas
    metrics_path = output_dir / "metrics.yaml"
    with open(metrics_path, "w") as f:
        yaml.dump(all_metrics, f, default_flow_style=False)
    logger.success(f"Métricas guardadas en {metrics_path}")

    # Guardar config del experimento
    config_path = output_dir / "experiment_config.yaml"
    experiment_config = {
        "model_type": model_type,
        "n_estimators": n_estimators,
        "use_smote": use_smote,
        "smote_strategy": smote_strategy,
        "calibration_method": calibration_method,
        "threshold": threshold,
        "features": available_features,
        "random_seed": cfg.random_seed,
        "train_split": "2018-2022",
        "val_split": "2023",
        "test_split": "2024",
    }
    with open(config_path, "w") as f:
        yaml.dump(experiment_config, f, default_flow_style=False)

    # Alerta si AUC-PR < 0.15 (objetivo mínimo para clase rara ~1-2%)
    if val_metrics["auc_pr"] < 0.15:
        logger.warning(
            f"AUC-PR en val ({val_metrics['auc_pr']:.4f}) por debajo del mínimo esperado (0.15). "
            "Considera ajustar hiperparámetros o revisar el dataset."
        )

    return all_metrics


@app.command()
def main(
    dataset_path: Annotated[
        Path,
        typer.Option("--dataset-path", help="Ruta al parquet M1"),
    ] = Path("data/processed/m1_dataset.parquet"),
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-o", help="Directorio de salida del modelo"),
    ] = Path("models_artifacts/m1"),
    model_type: Annotated[
        str,
        typer.Option("--model", help="Tipo de modelo: xgboost o lightgbm"),
    ] = "lightgbm",
    n_estimators: Annotated[
        int,
        typer.Option("--n-estimators", help="Número máximo de árboles"),
    ] = 500,
    no_smote: Annotated[
        bool,
        typer.Option("--no-smote", help="Desactivar SMOTE"),
    ] = False,
    smote_strategy: Annotated[
        float,
        typer.Option("--smote-strategy", help="Ratio SMOTE minoritaria/mayoritaria"),
    ] = 0.1,
    calibration_method: Annotated[
        str,
        typer.Option("--calibration", help="Método de calibración: sigmoid o isotonic"),
    ] = "isotonic",
    threshold: Annotated[
        float,
        typer.Option("--threshold", help="Umbral de clasificación"),
    ] = 0.05,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Nivel de logging"),
    ] = "INFO",
) -> None:
    """Entrena el modelo M1 de riesgo de incendio forestal."""
    logger.remove()
    logger.add(sys.stderr, level=log_level.upper())

    train_m1(
        dataset_path=dataset_path,
        output_dir=output_dir,
        model_type=model_type,
        n_estimators=n_estimators,
        use_smote=not no_smote,
        smote_strategy=smote_strategy,
        calibration_method=calibration_method,
        threshold=threshold,
    )


if __name__ == "__main__":
    app()

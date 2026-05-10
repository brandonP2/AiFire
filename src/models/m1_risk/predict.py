"""Inferencia del modelo M1: genera el mapa de riesgo de ignición para una fecha dada.

Dado un modelo entrenado y datos de entrada para un día específico, produce:
- Un DataFrame con probabilidad de ignición por celda (columna `risk_prob`).
- Un raster GeoTIFF opcional con el mapa de riesgo.

Uso:
    uv run python -m src.models.m1_risk.predict --date 2024-03-15
    uv run python -m src.models.m1_risk.predict --date 2024-03-15 --horizon 3
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Annotated

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import typer
from loguru import logger

from src.data.config import Settings
from src.data.config import settings as default_settings
from src.models.m1_risk.train import FEATURE_COLS

app = typer.Typer(add_completion=False)


def load_model(model_path: Path) -> object:
    """Carga el modelo M1 calibrado desde disco.

    Args:
        model_path: Ruta al archivo .joblib del modelo.

    Returns:
        Modelo cargado con método predict_proba.

    Raises:
        FileNotFoundError: Si el archivo no existe.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo no encontrado: {model_path}. "
            "Ejecuta primero: uv run python -m src.models.m1_risk.train"
        )
    model = joblib.load(model_path)
    logger.info(f"Modelo cargado desde {model_path}")
    return model


def prepare_features_for_date(
    target_date: date,
    dataset_path: Path,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Extrae las features del dataset para una fecha específica.

    Args:
        target_date: Fecha de predicción.
        dataset_path: Ruta al parquet M1.
        feature_cols: Lista de columnas de features a usar.

    Returns:
        DataFrame con columnas [cell_id] + feature_cols para la fecha dada.

    Raises:
        ValueError: Si la fecha no existe en el dataset.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset no encontrado: {dataset_path}")

    df = pd.read_parquet(dataset_path)
    df["date"] = pd.to_datetime(df["date"]).dt.date

    day_df = df[df["date"] == target_date].copy()
    if day_df.empty:
        raise ValueError(
            f"No hay datos en el dataset para la fecha {target_date}. "
            "Verifica que el dataset cubra ese período."
        )

    available_features = [c for c in feature_cols if c in day_df.columns]
    result = day_df[["cell_id", *available_features]].copy()

    # Imputar NaN con mediana del dataset para inferencia (simple, robusto)
    for col in available_features:
        if result[col].isna().any():
            median_val = df[col].median()
            result[col] = result[col].fillna(median_val)

    return result


def predict_risk(
    model,
    features_df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Genera probabilidades de riesgo de ignición por celda.

    Args:
        model: Modelo calibrado con predict_proba.
        features_df: DataFrame con columnas cell_id + features.
        feature_cols: Columnas de features en el orden esperado por el modelo.

    Returns:
        DataFrame con columnas [cell_id, risk_prob].
    """
    available = [c for c in feature_cols if c in features_df.columns]
    x_arr = features_df[available].values.astype(np.float32)

    proba = model.predict_proba(x_arr)[:, 1]

    return pd.DataFrame(
        {
            "cell_id": features_df["cell_id"].values,
            "risk_prob": proba,
        }
    )


def export_risk_raster(
    risk_df: pd.DataFrame,
    grid_path: Path,
    output_path: Path,
    target_date: date,
) -> None:
    """Exporta el mapa de riesgo como GeoTIFF.

    Args:
        risk_df: DataFrame con columnas [cell_id, risk_prob].
        grid_path: Ruta al GeoPackage de la grilla.
        output_path: Ruta del GeoTIFF de salida.
        target_date: Fecha de predicción (para metadatos).
    """
    import rasterio
    import rasterio.transform

    grid_gdf = gpd.read_file(grid_path, layer="peten_grid")
    merged = grid_gdf.merge(risk_df, on="cell_id", how="left")

    # Calcular el extent de la grilla en EPSG:32616
    bounds = merged.total_bounds  # (minx, miny, maxx, maxy)
    res_m = 1000.0  # 1 km

    n_cols = round((bounds[2] - bounds[0]) / res_m)
    n_rows = round((bounds[3] - bounds[1]) / res_m)

    transform = rasterio.transform.from_bounds(
        west=bounds[0], south=bounds[1],
        east=bounds[2], north=bounds[3],
        width=n_cols, height=n_rows,
    )

    # Crear raster vacío
    raster = np.full((n_rows, n_cols), np.nan, dtype=np.float32)

    # Rellenar con probabilidades
    for _, row in merged.iterrows():
        if pd.isna(row.get("risk_prob")):
            continue
        centroid = row.geometry.centroid
        from rasterio.transform import rowcol
        r, c = rowcol(transform, centroid.x, centroid.y)
        if 0 <= r < n_rows and 0 <= c < n_cols:
            raster[r, c] = row["risk_prob"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_path, "w",
        driver="GTiff",
        height=n_rows, width=n_cols,
        count=1,
        dtype="float32",
        crs="EPSG:32616",
        transform=transform,
        nodata=np.nan,
    ) as dst:
        dst.write(raster, 1)
        dst.update_tags(date=str(target_date), model="M1_risk")

    logger.success(f"Raster de riesgo guardado en {output_path}")


def run_prediction(
    target_date: date,
    horizon_days: int = 1,
    cfg: Settings | None = None,
    model_type: str = "lightgbm",
    export_raster: bool = False,
) -> pd.DataFrame:
    """Genera el mapa de riesgo para una fecha dada.

    Args:
        target_date: Fecha de predicción.
        horizon_days: Horizonte en días (1, 3 o 7).
        cfg: Configuración del proyecto.
        model_type: Tipo de modelo a usar.
        export_raster: Si exportar también el GeoTIFF.

    Returns:
        DataFrame con columnas [cell_id, risk_prob].
    """
    cfg = cfg or default_settings
    model_path = cfg.project_root / "models_artifacts" / "m1" / f"m1_{model_type}_calibrated.joblib"
    dataset_path = cfg.data_processed / "m1_dataset.parquet"
    grid_path = cfg.data_interim / "peten_grid.gpkg"

    model = load_model(model_path)

    # Para horizontes > 1 día, usar la fecha más lejana disponible
    prediction_date = target_date + timedelta(days=horizon_days - 1)
    logger.info(f"Predicción para {prediction_date} (horizonte {horizon_days}d desde {target_date})")

    features_df = prepare_features_for_date(prediction_date, dataset_path, FEATURE_COLS)
    risk_df = predict_risk(model, features_df, FEATURE_COLS)

    logger.info(
        f"Riesgo generado: {len(risk_df):,} celdas | "
        f"p50={risk_df['risk_prob'].median():.4f} | "
        f"p95={risk_df['risk_prob'].quantile(0.95):.4f}"
    )

    if export_raster:
        raster_path = (
            cfg.project_root
            / "reports"
            / f"risk_map_{prediction_date.isoformat()}_h{horizon_days}d.tif"
        )
        export_risk_raster(risk_df, grid_path, raster_path, prediction_date)

    return risk_df


@app.command()
def main(
    target_date: Annotated[
        str,
        typer.Option("--date", help="Fecha de predicción (YYYY-MM-DD)"),
    ] = "2024-03-15",
    horizon: Annotated[
        int,
        typer.Option("--horizon", help="Horizonte de predicción en días (1, 3 o 7)"),
    ] = 1,
    model_type: Annotated[
        str,
        typer.Option("--model", help="Tipo de modelo: xgboost o lightgbm"),
    ] = "lightgbm",
    export_raster: Annotated[
        bool,
        typer.Option("--export-raster", help="Exportar mapa de riesgo como GeoTIFF"),
    ] = False,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Nivel de logging"),
    ] = "INFO",
) -> None:
    """Genera el mapa de riesgo de incendio para una fecha dada."""
    logger.remove()
    logger.add(sys.stderr, level=log_level.upper())

    risk_df = run_prediction(
        target_date=date.fromisoformat(target_date),
        horizon_days=horizon,
        model_type=model_type,
        export_raster=export_raster,
    )

    top10 = risk_df.nlargest(10, "risk_prob")
    logger.info("Top 10 celdas de mayor riesgo:")
    for _, row in top10.iterrows():
        logger.info(f"  {row['cell_id']}: {row['risk_prob']:.4f}")


if __name__ == "__main__":
    app()

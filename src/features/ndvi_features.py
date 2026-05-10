"""Features de NDVI por celda: interpolación de composiciones MODIS 16-días a diario.

MODIS MOD13Q1 produce una composición cada 16 días a 250m de resolución.
Este módulo:
1. Carga los GeoTIFFs de NDVI (data/interim/ndvi/ndvi_{YYYY-MM-DD}.tif).
2. Calcula la media zonal por celda de 1km (reproyección o muestreo del centroide).
3. Interpola linealmente a frecuencia diaria.
4. Calcula lags de 7 y 14 días del NDVI.

Output:
    data/interim/ndvi/ndvi_features_{year}.parquet — una fila por (cell_id, date)

Uso:
    uv run python -m src.features.ndvi_features
    uv run python -m src.features.ndvi_features --start 2018-01-01 --end 2024-12-31
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Annotated

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import typer
from loguru import logger
from rasterio.transform import rowcol

from src.data.config import Settings
from src.data.config import settings as default_settings

app = typer.Typer(add_completion=False)

NDVI_SCALE = 0.0001  # factor de escala MODIS: valor_entero * 0.0001 = NDVI real
NDVI_NODATA = -3000  # valor de no-dato en MODIS MOD13Q1


# ---------------------------------------------------------------------------
# Extracción de NDVI por celda desde GeoTIFF
# ---------------------------------------------------------------------------


def _extract_ndvi_from_tif(
    tif_path: Path,
    cell_ids: np.ndarray,
    centroids_xy: np.ndarray,
    target_epsg: int = 32616,
) -> pd.Series:
    """Muestrea el NDVI en los centroides de las celdas desde un GeoTIFF MODIS.

    Si el raster no está en EPSG:32616, reproyecta los puntos al CRS del raster
    en lugar de reproyectar el raster (más rápido para muestreo puntual).

    Args:
        tif_path: Ruta al GeoTIFF NDVI.
        cell_ids: Array (N,) de cell_id.
        centroids_xy: Array (N, 2) de coordenadas (easting, northing) en EPSG:32616.
        target_epsg: EPSG de las coordenadas de entrada (normalmente 32616).

    Returns:
        Series con cell_id como índice y valores NDVI en [-1, 1] (NaN si no hay dato).
    """
    from pyproj import Transformer

    with rasterio.open(tif_path) as src:
        raster_epsg = src.crs.to_epsg() if src.crs else None
        nodata = src.nodata if src.nodata is not None else NDVI_NODATA

        if raster_epsg != target_epsg:
            transformer = Transformer.from_crs(
                f"EPSG:{target_epsg}", f"EPSG:{raster_epsg}", always_xy=True
            )
            xs_r, ys_r = transformer.transform(centroids_xy[:, 0], centroids_xy[:, 1])
        else:
            xs_r, ys_r = centroids_xy[:, 0], centroids_xy[:, 1]

        rows, cols = rowcol(src.transform, xs_r, ys_r)
        rows = np.asarray(rows, dtype=int)
        cols = np.asarray(cols, dtype=int)

        data = src.read(1).astype(float)
        valid_mask = (
            (rows >= 0) & (rows < data.shape[0]) & (cols >= 0) & (cols < data.shape[1])
        )

        values = np.full(len(cell_ids), np.nan)
        vr, vc = rows[valid_mask], cols[valid_mask]
        sampled = data[vr, vc]
        # Aplicar escala y filtrar no-datos
        sampled = np.where(sampled == nodata, np.nan, sampled * NDVI_SCALE)
        values[valid_mask] = sampled

    return pd.Series(values, index=cell_ids, name="ndvi")


# ---------------------------------------------------------------------------
# Descubrimiento de GeoTIFFs disponibles
# ---------------------------------------------------------------------------


def _discover_ndvi_tifs(ndvi_dir: Path) -> dict[date, Path]:
    """Encuentra todos los GeoTIFFs NDVI en el directorio y extrae sus fechas.

    Formato de nombre esperado: ndvi_YYYY-MM-DD.tif

    Args:
        ndvi_dir: Directorio con los archivos ndvi_*.tif.

    Returns:
        Diccionario {fecha_composición: ruta}.
    """
    result: dict[date, Path] = {}
    for tif in sorted(ndvi_dir.glob("ndvi_*.tif")):
        stem = tif.stem  # "ndvi_2023-03-14"
        date_str = stem.replace("ndvi_", "")
        try:
            d = date.fromisoformat(date_str)
            result[d] = tif
        except ValueError:
            logger.warning(f"No se pudo parsear fecha de {tif.name}, ignorando")
    return result


# ---------------------------------------------------------------------------
# Interpolación diaria
# ---------------------------------------------------------------------------


def interpolate_ndvi_to_daily(
    ndvi_composites: pd.DataFrame,
    start: date,
    end: date,
) -> pd.DataFrame:
    """Interpola linealmente el NDVI de composiciones 16-días a frecuencia diaria.

    Args:
        ndvi_composites: DataFrame con columnas [cell_id, date, ndvi].
            date corresponde a la fecha de inicio de cada composición 16-días.
        start: Primera fecha del rango diario.
        end: Última fecha del rango diario.

    Returns:
        DataFrame con columnas [cell_id, date, ndvi, ndvi_lag7, ndvi_lag14].
        ndvi_lag7 y ndvi_lag14 son NaN para los primeros días sin historia.
    """
    daily_dates = pd.date_range(start=start, end=end, freq="D").date

    pivot = ndvi_composites.pivot(index="date", columns="cell_id", values="ndvi")

    # Reindexar a diario e interpolar
    pivot.index = pd.DatetimeIndex(pivot.index)
    daily_index = pd.DatetimeIndex(daily_dates)
    pivot_full = pivot.reindex(daily_index).interpolate(method="time", limit_direction="both")

    # Volver a formato largo
    daily_df = pivot_full.stack(future_stack=True).reset_index()
    daily_df.columns = pd.Index(["date", "cell_id", "ndvi"])
    daily_df["date"] = daily_df["date"].dt.date

    # Calcular lags por celda (ordenados por fecha, luego shift)
    daily_df = daily_df.sort_values(["cell_id", "date"]).reset_index(drop=True)
    daily_df["ndvi_lag7"] = daily_df.groupby("cell_id")["ndvi"].shift(7)
    daily_df["ndvi_lag14"] = daily_df.groupby("cell_id")["ndvi"].shift(14)

    return daily_df


# ---------------------------------------------------------------------------
# Pipeline anual
# ---------------------------------------------------------------------------


def process_ndvi_year(
    year: int,
    grid_df: pd.DataFrame,
    centroids_xy: np.ndarray,
    ndvi_tifs: dict[date, Path],
    output_dir: Path,
    padding_days: int = 14,
) -> Path | None:
    """Procesa el NDVI de un año y genera el parquet de features diarias.

    Args:
        year: Año a procesar.
        grid_df: DataFrame con columnas cell_id, easting_m, northing_m.
        centroids_xy: Array (N, 2) de coordenadas UTM de los centroides.
        ndvi_tifs: Mapa {fecha: ruta_tif} de composiciones disponibles.
        output_dir: Directorio de salida.
        padding_days: Días extra al inicio del año para calcular lags correctamente.

    Returns:
        Ruta al parquet generado, o None si no hay datos.
    """
    output_path = output_dir / f"ndvi_features_{year}.parquet"
    if output_path.exists():
        logger.info(f"Ya existe: {output_path} — saltando")
        return output_path

    start_with_padding = date(year - 1 if padding_days > 0 else year, 12, 18) if padding_days >= 14 else date(year, 1, 1)
    start_year = date(year, 1, 1)
    end_year = date(year, 12, 31)

    # Seleccionar composiciones relevantes (con margen para el padding)
    margin_start = date(year - 1, 11, 1) if year > 2018 else date(year, 1, 1)
    relevant_tifs = {
        d: p
        for d, p in ndvi_tifs.items()
        if margin_start <= d <= end_year
    }

    if not relevant_tifs:
        logger.warning(f"Sin GeoTIFFs NDVI para el año {year}")
        return None

    logger.info(f"NDVI {year}: {len(relevant_tifs)} composiciones disponibles")

    cell_ids = grid_df["cell_id"].values
    composites_list: list[pd.DataFrame] = []

    for comp_date, tif_path in sorted(relevant_tifs.items()):
        ndvi_series = _extract_ndvi_from_tif(tif_path, cell_ids, centroids_xy)
        comp_df = pd.DataFrame(
            {
                "cell_id": cell_ids,
                "date": comp_date,
                "ndvi": ndvi_series.values,
            }
        )
        composites_list.append(comp_df)

    if not composites_list:
        return None

    composites_df = pd.concat(composites_list, ignore_index=True)

    # Interpolar a diario para el año completo (con padding al inicio)
    interp_df = interpolate_ndvi_to_daily(composites_df, start_with_padding, end_year)

    # Filtrar al año actual (sin el padding)
    result = interp_df[interp_df["date"] >= start_year].copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    logger.success(f"NDVI {year}: {len(result):,} filas → {output_path}")
    return output_path


def build_ndvi_features(
    cfg: Settings | None = None,
    start: date | None = None,
    end: date | None = None,
    grid_path: Path | None = None,
    output_dir: Path | None = None,
) -> list[Path]:
    """Procesa el NDVI de todos los años en el rango dado.

    Args:
        cfg: Configuración del proyecto.
        start: Fecha de inicio (inclusiva).
        end: Fecha de fin (inclusiva).
        grid_path: Ruta al GeoPackage de la grilla.
        output_dir: Directorio de salida.

    Returns:
        Lista de rutas a los parquets generados.
    """
    cfg = cfg or default_settings
    start = start or date.fromisoformat(cfg.train_start)
    end = end or date.fromisoformat(cfg.test_end)
    grid_path = grid_path or (cfg.data_interim / "peten_grid.gpkg")
    output_dir = output_dir or (cfg.data_interim / "ndvi")
    ndvi_dir = cfg.data_interim / "ndvi"

    if not grid_path.exists():
        raise FileNotFoundError(f"Grilla no encontrada: {grid_path}")

    logger.info(f"Cargando grilla desde {grid_path}...")
    grid_gdf = gpd.read_file(grid_path, layer="peten_grid")
    grid_df = grid_gdf[["cell_id", "easting_m", "northing_m"]].copy()
    centroids_xy = np.column_stack([grid_df["easting_m"].values, grid_df["northing_m"].values])

    ndvi_tifs = _discover_ndvi_tifs(ndvi_dir)
    logger.info(f"GeoTIFFs NDVI disponibles: {len(ndvi_tifs)}")

    years = range(start.year, end.year + 1)
    paths: list[Path] = []
    for year in years:
        p = process_ndvi_year(year, grid_df, centroids_xy, ndvi_tifs, output_dir)
        if p is not None:
            paths.append(p)

    logger.success(f"NDVI procesado: {len(paths)}/{len(list(years))} años")
    return paths


@app.command()
def main(
    start: Annotated[
        str,
        typer.Option("--start", help="Fecha inicio (YYYY-MM-DD)"),
    ] = "2018-01-01",
    end: Annotated[
        str,
        typer.Option("--end", help="Fecha fin (YYYY-MM-DD)"),
    ] = "2024-12-31",
    grid_path: Annotated[
        Path,
        typer.Option("--grid-path", help="Ruta al GeoPackage de la grilla"),
    ] = Path("data/interim/peten_grid.gpkg"),
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Nivel de logging"),
    ] = "INFO",
) -> None:
    """Genera features de NDVI diarias por celda para todo el período."""
    logger.remove()
    logger.add(sys.stderr, level=log_level.upper())
    build_ndvi_features(
        start=date.fromisoformat(start),
        end=date.fromisoformat(end),
        grid_path=grid_path,
    )


if __name__ == "__main__":
    app()

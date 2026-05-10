"""Features climáticas diarias por celda: NASA POWER + FWI iterativo.

NASA POWER entrega una grilla de puntos ~0.5 x 0.5 grados. Este módulo:
1. Carga los parquets anuales de POWER (data/raw/power/power_{year}.parquet).
2. Asigna a cada celda el punto POWER más cercano (vecino más próximo en EPSG:4326).
3. Calcula el FWI canadiense de forma iterativa día a día, vectorizado sobre celdas.

Output:
    data/interim/climate/climate_features_{year}.parquet — una fila por (cell_id, date)

Uso:
    uv run python -m src.features.climate_features
    uv run python -m src.features.climate_features --start 2018-01-01 --end 2024-12-31
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer
from loguru import logger
from scipy.spatial import cKDTree

from src.data.config import Settings
from src.data.config import settings as default_settings
from src.features.fwi import bui, dc, dmc, ffmc, fwi, isi

app = typer.Typer(add_completion=False)

# Valores iniciales estándar FWI (inicio de temporada, suelos húmedos)
_FFMC0 = 85.0
_DMC0 = 6.0
_DC0 = 15.0


# ---------------------------------------------------------------------------
# Asignación POWER → celdas
# ---------------------------------------------------------------------------


def assign_power_points_to_cells(
    cell_ids: np.ndarray,
    cell_lons: np.ndarray,
    cell_lats: np.ndarray,
    power_lons: np.ndarray,
    power_lats: np.ndarray,
) -> np.ndarray:
    """Para cada celda, encuentra el índice del punto POWER más cercano.

    Usa KD-tree sobre coordenadas geográficas (grados). Suficientemente preciso
    para la resolución de POWER (~0.5°).

    Args:
        cell_ids: Array de cell_id (N,).
        cell_lons: Longitud del centroide de cada celda (N,).
        cell_lats: Latitud del centroide de cada celda (N,).
        power_lons: Longitudes únicas de los puntos POWER (M,).
        power_lats: Latitudes únicas de los puntos POWER (M,).

    Returns:
        Array (N,) de índices en el espacio de puntos POWER (filas de power_coords).
    """
    power_coords = np.column_stack([power_lons, power_lats])
    cell_coords = np.column_stack([cell_lons, cell_lats])
    tree = cKDTree(power_coords)
    _, indices = tree.query(cell_coords, k=1)
    return indices


# ---------------------------------------------------------------------------
# FWI iterativo
# ---------------------------------------------------------------------------


def compute_fwi_series(
    climate_by_cell: pd.DataFrame,
    cell_ids_ordered: np.ndarray,
    ffmc0: float = _FFMC0,
    dmc0: float = _DMC0,
    dc0: float = _DC0,
) -> pd.DataFrame:
    """Calcula el FWI canadiense iterando sobre fechas, vectorizado sobre celdas.

    Args:
        climate_by_cell: DataFrame con columnas [cell_id, date, T2M, RH2M, WS10M,
            PRECTOTCORR, month]. Ordenado por fecha.
        cell_ids_ordered: Array de cell_id en el orden en que aparecen en los arrays
            del estado FWI (mismo orden que el pivoteo interno).
        ffmc0: FFMC inicial (inicio de serie).
        dmc0: DMC inicial.
        dc0: DC inicial.

    Returns:
        DataFrame con columnas [cell_id, date, ffmc_val, dmc_val, dc_val,
            isi_val, bui_val, fwi_val].
    """
    dates = climate_by_cell["date"].sort_values().unique()
    n_cells = len(cell_ids_ordered)

    # Pivotear para tener arrays (n_cells,) por variable por fecha
    pivot_temp = climate_by_cell.pivot(index="date", columns="cell_id", values="T2M")
    pivot_rh = climate_by_cell.pivot(index="date", columns="cell_id", values="RH2M")
    pivot_wind = climate_by_cell.pivot(index="date", columns="cell_id", values="WS10M")
    pivot_rain = climate_by_cell.pivot(index="date", columns="cell_id", values="PRECTOTCORR")
    pivot_month = climate_by_cell.pivot(index="date", columns="cell_id", values="month")

    # Reindexar columnas al orden esperado
    pivot_temp = pivot_temp.reindex(columns=cell_ids_ordered)
    pivot_rh = pivot_rh.reindex(columns=cell_ids_ordered)
    pivot_wind = pivot_wind.reindex(columns=cell_ids_ordered)
    pivot_rain = pivot_rain.reindex(columns=cell_ids_ordered)
    pivot_month = pivot_month.reindex(columns=cell_ids_ordered)

    # Estado inicial del FWI
    ffmc_prev = np.full(n_cells, ffmc0)
    dmc_prev = np.full(n_cells, dmc0)
    dc_prev = np.full(n_cells, dc0)

    records: list[dict] = []

    for dt in dates:
        temp = pivot_temp.loc[dt].values.astype(float)
        rh = pivot_rh.loc[dt].values.astype(float)
        wind = pivot_wind.loc[dt].values.astype(float)
        rain = pivot_rain.loc[dt].values.astype(float)
        month_arr = pivot_month.loc[dt].values.astype(float)

        # Usar la moda del mes (todos los valores deberían ser iguales)
        month_scalar = int(np.nanmedian(month_arr))

        # Reemplazar NaN en inputs con valores seguros para que el FWI no explote
        temp_safe = np.where(np.isnan(temp), 25.0, temp)
        rh_safe = np.clip(np.where(np.isnan(rh), 50.0, rh), 1.0, 99.0)
        wind_safe = np.where(np.isnan(wind), 10.0, wind)
        rain_safe = np.where(np.isnan(rain), 0.0, rain)

        ffmc_cur = ffmc(temp_safe, rh_safe, wind_safe, rain_safe, ffmc_prev)
        dmc_cur = dmc(temp_safe, rh_safe, rain_safe, month_scalar, dmc_prev)
        dc_cur = dc(temp_safe, rain_safe, month_scalar, dc_prev)
        isi_cur = isi(wind_safe, ffmc_cur)
        bui_cur = bui(dmc_cur, dc_cur)
        fwi_cur = fwi(isi_cur, bui_cur)

        # Marcar NaN donde el input climático original era NaN (dato faltante real)
        missing_mask = np.isnan(temp) | np.isnan(rh) | np.isnan(wind) | np.isnan(rain)
        fwi_masked = np.where(missing_mask, np.nan, fwi_cur)

        records.append(
            pd.DataFrame(
                {
                    "cell_id": cell_ids_ordered,
                    "date": dt,
                    "ffmc_val": ffmc_cur,
                    "dmc_val": dmc_cur,
                    "dc_val": dc_cur,
                    "isi_val": isi_cur,
                    "bui_val": bui_cur,
                    "fwi_val": fwi_masked,
                }
            )
        )

        ffmc_prev = ffmc_cur
        dmc_prev = dmc_cur
        dc_prev = dc_cur

    return pd.concat(records, ignore_index=True)


# ---------------------------------------------------------------------------
# Pipeline anual
# ---------------------------------------------------------------------------


def process_climate_year(
    year: int,
    grid_df: pd.DataFrame,
    power_dir: Path,
    output_dir: Path,
) -> Path | None:
    """Procesa los datos POWER de un año y calcula el FWI para todas las celdas.

    Args:
        year: Año a procesar.
        grid_df: DataFrame con columnas cell_id, lon_center, lat_center.
        power_dir: Directorio con archivos power_{year}.parquet.
        output_dir: Directorio de salida.

    Returns:
        Ruta al parquet generado, o None si el año no tiene datos.
    """
    power_path = power_dir / f"power_{year}.parquet"
    if not power_path.exists():
        logger.warning(f"Sin datos POWER para {year}: {power_path}")
        return None

    output_path = output_dir / f"climate_features_{year}.parquet"
    if output_path.exists():
        logger.info(f"Ya existe: {output_path} — saltando")
        return output_path

    logger.info(f"Procesando clima {year}...")
    power_df = pd.read_parquet(power_path)

    # Validar columnas necesarias
    required = {"lat", "lon", "date", "T2M", "RH2M", "WS10M", "PRECTOTCORR"}
    missing_cols = required - set(power_df.columns)
    if missing_cols:
        logger.error(f"Columnas faltantes en POWER {year}: {missing_cols}")
        return None

    power_df["date"] = pd.to_datetime(power_df["date"]).dt.date
    power_df["month"] = pd.to_datetime(power_df["date"]).dt.month

    # Puntos únicos de la grilla POWER
    power_points = (
        power_df[["lat", "lon"]].drop_duplicates().reset_index(drop=True)
    )
    power_lons = power_points["lon"].values
    power_lats = power_points["lat"].values

    # Asignar punto POWER a cada celda (lookup estático por año)
    cell_ids = grid_df["cell_id"].values
    cell_lons = grid_df["lon_center"].values
    cell_lats = grid_df["lat_center"].values

    indices = assign_power_points_to_cells(
        cell_ids, cell_lons, cell_lats, power_lons, power_lats
    )
    # Crear mapa: cell_id → (power_lat, power_lon)
    cell_to_power = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "power_lat": power_lats[indices],
            "power_lon": power_lons[indices],
        }
    )

    # Expandir: por cada (cell_id, date), obtener el clima del punto POWER asignado
    climate_expanded = cell_to_power.merge(
        power_df,
        left_on=["power_lat", "power_lon"],
        right_on=["lat", "lon"],
        how="left",
    )[["cell_id", "date", "T2M", "RH2M", "WS10M", "PRECTOTCORR", "month"]]

    # Convertir velocidad del viento: POWER da m/s, FWI necesita km/h
    # Creamos una vista separada para el FWI (sin modificar climate_expanded)
    climate_for_fwi = climate_expanded.copy()
    climate_for_fwi["WS10M"] = climate_for_fwi["WS10M"] * 3.6

    # Calcular FWI
    logger.info(f"Calculando FWI para {year} ({len(cell_ids):,} celdas)...")
    fwi_df = compute_fwi_series(
        climate_for_fwi,
        cell_ids_ordered=cell_ids,
    )

    # Unir clima original (WS10M en m/s) + FWI
    result = climate_expanded.merge(fwi_df, on=["cell_id", "date"], how="left")

    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    logger.success(f"Clima {year}: {len(result):,} filas → {output_path}")
    return output_path


def build_climate_features(
    cfg: Settings | None = None,
    start: date | None = None,
    end: date | None = None,
    grid_path: Path | None = None,
    output_dir: Path | None = None,
) -> list[Path]:
    """Procesa el clima de todos los años en el rango dado.

    Args:
        cfg: Configuración del proyecto.
        start: Fecha de inicio (inclusiva).
        end: Fecha de fin (inclusiva).
        grid_path: Ruta al parquet/GeoPackage de la grilla.
        output_dir: Directorio de salida.

    Returns:
        Lista de rutas a los parquets generados.
    """
    import geopandas as gpd

    cfg = cfg or default_settings
    start = start or date.fromisoformat(cfg.train_start)
    end = end or date.fromisoformat(cfg.test_end)
    grid_path = grid_path or (cfg.data_interim / "peten_grid.gpkg")
    output_dir = output_dir or (cfg.data_interim / "climate")
    power_dir = cfg.data_raw / "power"

    if not grid_path.exists():
        raise FileNotFoundError(f"Grilla no encontrada: {grid_path}")

    logger.info(f"Cargando grilla desde {grid_path}...")
    grid_gdf = gpd.read_file(grid_path, layer="peten_grid")
    grid_df = grid_gdf[["cell_id", "lon_center", "lat_center"]].copy()

    years = range(start.year, end.year + 1)
    paths: list[Path] = []
    for year in years:
        p = process_climate_year(year, grid_df, power_dir, output_dir)
        if p is not None:
            paths.append(p)

    logger.success(f"Clima procesado: {len(paths)}/{len(list(years))} años")
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
    """Calcula features climáticas y FWI por celda/día para todo el período."""
    logger.remove()
    logger.add(sys.stderr, level=log_level.upper())
    build_climate_features(
        start=date.fromisoformat(start),
        end=date.fromisoformat(end),
        grid_path=grid_path,
    )


if __name__ == "__main__":
    app()

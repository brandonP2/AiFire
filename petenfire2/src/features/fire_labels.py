"""Etiquetas de incendio por celda/día a partir de NASA FIRMS.

Para cada celda de la grilla del Petén y cada fecha en el rango, determina si
ocurrió al menos un foco activo FIRMS (VIIRS o MODIS) dentro de la celda.

Proceso:
1. Carga los CSV/parquets de FIRMS de data/raw/firms/.
2. Convierte focos a GeoDataFrame (puntos en EPSG:4326 → EPSG:32616).
3. Hace sjoin espacial con la grilla para asignar cell_id a cada foco.
4. Agrega por (cell_id, date) → fire_occurred = True si ≥ 1 foco.

Output:
    data/interim/fire_labels/fire_labels_{year}.parquet — (cell_id, date, fire_occurred)

Uso:
    uv run python -m src.features.fire_labels
    uv run python -m src.features.fire_labels --start 2018-01-01 --end 2024-12-31
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Annotated

import geopandas as gpd
import pandas as pd
import typer
from loguru import logger

from src.data.config import Settings
from src.data.config import settings as default_settings

app = typer.Typer(add_completion=False)

# Nivel de confianza mínimo para incluir un foco VIIRS
# MODIS usa strings ('low', 'nominal', 'high'); VIIRS usa enteros (0-100)
VIIRS_MIN_CONFIDENCE = 30


# ---------------------------------------------------------------------------
# Carga de datos FIRMS
# ---------------------------------------------------------------------------


def _load_firms_year(firms_dir: Path, year: int) -> pd.DataFrame | None:
    """Carga todos los archivos FIRMS disponibles para un año dado.

    Busca archivos con patrones: firms_{source}_{year}*.parquet o *.csv

    Args:
        firms_dir: Directorio con archivos FIRMS.
        year: Año a cargar.

    Returns:
        DataFrame unificado con columnas latitude, longitude, acq_date, confidence,
        o None si no hay archivos.
    """
    dfs: list[pd.DataFrame] = []

    for pattern in ("*.parquet", "*.csv"):
        for p in sorted(firms_dir.glob(pattern)):
            try:
                if p.suffix == ".parquet":
                    df = pd.read_parquet(p)
                else:
                    df = pd.read_csv(p, low_memory=False)
                dfs.append(df)
                logger.debug(f"Cargado {p.name}: {len(df):,} focos")
            except Exception as exc:
                logger.warning(f"Error cargando {p.name}: {exc}")

    if not dfs:
        return None

    combined = pd.concat(dfs, ignore_index=True)

    # Normalizar nombre de columna de brillo (MODIS usa 'brightness' → 'bright_ti4')
    if "brightness" in combined.columns and "bright_ti4" not in combined.columns:
        combined = combined.rename(columns={"brightness": "bright_ti4"})

    # Normalizar fecha
    combined["acq_date"] = pd.to_datetime(combined["acq_date"]).dt.date

    # Filtrar por confianza mínima
    if "confidence" in combined.columns:
        # VIIRS: entero; MODIS: string
        is_numeric = pd.to_numeric(combined["confidence"], errors="coerce").notna()
        viirs_mask = is_numeric & (
            pd.to_numeric(combined["confidence"], errors="coerce") >= VIIRS_MIN_CONFIDENCE
        )
        modis_mask = (~is_numeric) & (
            combined["confidence"].astype(str).str.lower().isin(["nominal", "high", "n", "h"])
        )
        combined = combined[viirs_mask | modis_mask].copy()

    logger.info(f"FIRMS {year}: {len(combined):,} focos (tras filtro de confianza)")
    return combined


# ---------------------------------------------------------------------------
# Asignación de focos a celdas
# ---------------------------------------------------------------------------


def assign_fires_to_grid(
    firms_df: pd.DataFrame,
    grid_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Asigna cada foco FIRMS a la celda de la grilla que lo contiene.

    Args:
        firms_df: DataFrame con columnas latitude, longitude, acq_date.
        grid_gdf: GeoDataFrame de la grilla (EPSG:32616).

    Returns:
        DataFrame con columnas cell_id, date, fire_occurred.
        Incluye todas las combinaciones (cell_id, date) con fire_occurred=True.
    """
    if firms_df.empty:
        return pd.DataFrame(columns=["cell_id", "date", "fire_occurred"])

    # Crear GeoDataFrame de focos en EPSG:4326
    fire_gdf = gpd.GeoDataFrame(
        firms_df[["latitude", "longitude", "acq_date"]].copy(),
        geometry=gpd.points_from_xy(firms_df["longitude"], firms_df["latitude"]),
        crs="EPSG:4326",
    ).to_crs(grid_gdf.crs)

    # Spatial join: foco → celda contenedora
    joined = gpd.sjoin(
        fire_gdf[["acq_date", "geometry"]],
        grid_gdf[["cell_id", "geometry"]],
        how="inner",
        predicate="within",
    )

    if joined.empty:
        logger.warning("sjoin no encontró focos dentro de la grilla")
        return pd.DataFrame(columns=["cell_id", "date", "fire_occurred"])

    # Agregar: ¿hubo al menos un foco en (cell_id, date)?
    labels = (
        joined.rename(columns={"acq_date": "date"})
        .groupby(["cell_id", "date"])
        .size()
        .reset_index(name="n_fires")
    )
    labels["fire_occurred"] = True
    labels = labels.drop(columns=["n_fires"])

    return labels


# ---------------------------------------------------------------------------
# Pipeline anual
# ---------------------------------------------------------------------------


def process_fire_labels_year(
    year: int,
    grid_gdf: gpd.GeoDataFrame,
    firms_dir: Path,
    output_dir: Path,
    all_dates: list[date] | None = None,
) -> Path | None:
    """Genera las etiquetas de incendio para un año.

    Args:
        year: Año a procesar.
        grid_gdf: GeoDataFrame de la grilla.
        firms_dir: Directorio con archivos FIRMS.
        output_dir: Directorio de salida.
        all_dates: Lista de fechas del año (si None, se usa el año completo).

    Returns:
        Ruta al parquet generado, o None si no hay datos.
    """
    output_path = output_dir / f"fire_labels_{year}.parquet"
    if output_path.exists():
        logger.info(f"Ya existe: {output_path} — saltando")
        return output_path

    firms_df = _load_firms_year(firms_dir, year)
    if firms_df is None:
        logger.warning(f"Sin datos FIRMS para {year}")
        return None

    # Filtrar al año correcto
    firms_year = firms_df[
        firms_df["acq_date"].apply(lambda d: d.year) == year
    ].copy()

    labels = assign_fires_to_grid(firms_year, grid_gdf)

    output_dir.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(output_path, index=False)

    n_fire_cells = len(labels)
    logger.success(f"FIRMS {year}: {n_fire_cells:,} pares (celda, día) con fuego → {output_path}")
    return output_path


def build_fire_labels(
    cfg: Settings | None = None,
    start: date | None = None,
    end: date | None = None,
    grid_path: Path | None = None,
    output_dir: Path | None = None,
) -> list[Path]:
    """Genera etiquetas de incendio para todos los años en el rango.

    Args:
        cfg: Configuración del proyecto.
        start: Fecha de inicio.
        end: Fecha de fin.
        grid_path: Ruta al GeoPackage de la grilla.
        output_dir: Directorio de salida.

    Returns:
        Lista de rutas a los parquets generados.
    """
    cfg = cfg or default_settings
    start = start or date.fromisoformat(cfg.train_start)
    end = end or date.fromisoformat(cfg.test_end)
    grid_path = grid_path or (cfg.data_interim / "peten_grid.gpkg")
    output_dir = output_dir or (cfg.data_interim / "fire_labels")
    firms_dir = cfg.data_raw / "firms"

    if not grid_path.exists():
        raise FileNotFoundError(f"Grilla no encontrada: {grid_path}")

    logger.info(f"Cargando grilla desde {grid_path}...")
    grid_gdf = gpd.read_file(grid_path, layer="peten_grid")

    years = range(start.year, end.year + 1)
    paths: list[Path] = []
    for year in years:
        p = process_fire_labels_year(year, grid_gdf, firms_dir, output_dir)
        if p is not None:
            paths.append(p)

    logger.success(f"Etiquetas FIRMS: {len(paths)}/{len(list(years))} años")
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
    """Genera etiquetas de incendio (fire_occurred) por celda y día."""
    logger.remove()
    logger.add(sys.stderr, level=log_level.upper())
    build_fire_labels(
        start=date.fromisoformat(start),
        end=date.fromisoformat(end),
        grid_path=grid_path,
    )


if __name__ == "__main__":
    app()

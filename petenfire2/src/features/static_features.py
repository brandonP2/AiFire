"""Features estáticas por celda: topografía, distancias y áreas protegidas.

Inputs esperados (todos bajo data/interim/):
    topography/srtm_peten.tif    — elevación (m), EPSG:32616
    topography/slope_peten.tif   — pendiente (°), EPSG:32616
    topography/aspect_peten.tif  — aspecto (°), EPSG:32616
    static/roads_peten.gpkg      — caminos OSM
    static/settlements_peten.gpkg — poblados OSM
    static/protected_areas_peten.gpkg — áreas protegidas

Output:
    data/processed/static_features.parquet — una fila por cell_id

Uso:
    uv run python -m src.features.static_features
    uv run python -m src.features.static_features --grid-path data/interim/peten_grid.gpkg
"""

from __future__ import annotations

import sys
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


# ---------------------------------------------------------------------------
# Topografía
# ---------------------------------------------------------------------------


def _sample_raster_at_centroids(
    raster_path: Path,
    centroids_xy: np.ndarray,
) -> np.ndarray:
    """Muestrea un raster en las coordenadas dadas (en el CRS del raster).

    Args:
        raster_path: Ruta al GeoTIFF.
        centroids_xy: Array (N, 2) de coordenadas (x, y) en el CRS del raster.

    Returns:
        Array 1D de valores muestreados. NaN donde no hay dato.
    """
    with rasterio.open(raster_path) as src:
        nodata = src.nodata
        rows, cols = rowcol(src.transform, centroids_xy[:, 0], centroids_xy[:, 1])
        rows = np.asarray(rows, dtype=int)
        cols = np.asarray(cols, dtype=int)

        # Leer la banda completa y muestrear — eficiente para rasters pequeños
        data = src.read(1).astype(float)

        valid_mask = (
            (rows >= 0) & (rows < data.shape[0]) & (cols >= 0) & (cols < data.shape[1])
        )

        values = np.full(len(centroids_xy), np.nan)
        valid_rows = rows[valid_mask]
        valid_cols = cols[valid_mask]
        sampled = data[valid_rows, valid_cols]

        if nodata is not None:
            sampled = np.where(sampled == nodata, np.nan, sampled)

        values[valid_mask] = sampled

    return values


def compute_topo_features(
    grid_gdf: gpd.GeoDataFrame,
    topo_dir: Path,
) -> pd.DataFrame:
    """Extrae features topográficas por celda muestreando los rasters SRTM.

    Args:
        grid_gdf: GeoDataFrame de la grilla (EPSG:32616).
        topo_dir: Directorio con srtm_peten.tif, slope_peten.tif, aspect_peten.tif.

    Returns:
        DataFrame con columnas cell_id, elevation_m, slope_deg, aspect_deg.

    Raises:
        FileNotFoundError: Si algún raster SRTM no existe.
    """
    for name in ("srtm_peten.tif", "slope_peten.tif", "aspect_peten.tif"):
        p = topo_dir / name
        if not p.exists():
            raise FileNotFoundError(f"Raster SRTM no encontrado: {p}")

    centroids = grid_gdf.geometry.centroid
    xy = np.column_stack([centroids.x.values, centroids.y.values])

    logger.info("Muestreando elevación, pendiente y aspecto...")
    elevation = _sample_raster_at_centroids(topo_dir / "srtm_peten.tif", xy)
    slope = _sample_raster_at_centroids(topo_dir / "slope_peten.tif", xy)
    aspect = _sample_raster_at_centroids(topo_dir / "aspect_peten.tif", xy)

    df = pd.DataFrame(
        {
            "cell_id": grid_gdf["cell_id"].values,
            "elevation_m": elevation,
            "slope_deg": slope,
            "aspect_deg": aspect,
        }
    )

    n_missing = df[["elevation_m", "slope_deg", "aspect_deg"]].isna().any(axis=1).sum()
    if n_missing > 0:
        logger.warning(f"{n_missing} celdas sin datos topográficos (NaN)")

    logger.info(f"Topografía: {len(df):,} celdas procesadas")
    return df


# ---------------------------------------------------------------------------
# Distancias a caminos y poblados
# ---------------------------------------------------------------------------


def compute_distance_features(
    grid_gdf: gpd.GeoDataFrame,
    roads_path: Path,
    settlements_path: Path,
) -> pd.DataFrame:
    """Calcula la distancia mínima de cada celda a caminos y poblados (km).

    Usa sjoin_nearest de GeoPandas (requiere shapely ≥ 2.0 + PyGEOS o GEOS 3.10+).

    Args:
        grid_gdf: GeoDataFrame de la grilla (EPSG:32616).
        roads_path: GeoPackage de caminos OSM.
        settlements_path: GeoPackage de poblados OSM.

    Returns:
        DataFrame con columnas cell_id, dist_roads_km, dist_settlements_km.

    Raises:
        FileNotFoundError: Si alguna capa no existe.
    """
    for p in (roads_path, settlements_path):
        if not p.exists():
            raise FileNotFoundError(f"Capa estática no encontrada: {p}")

    centroids_gdf = grid_gdf[["cell_id", "geometry"]].copy()
    centroids_gdf["geometry"] = centroids_gdf.geometry.centroid

    logger.info("Calculando distancia a caminos...")
    roads_gdf = gpd.read_file(roads_path).to_crs(grid_gdf.crs)
    joined_roads = gpd.sjoin_nearest(
        centroids_gdf, roads_gdf[["geometry"]], how="left", distance_col="dist_roads_m"
    )
    # sjoin_nearest puede duplicar filas si hay empates; quedarse con la menor distancia
    roads_dist = (
        joined_roads.groupby("cell_id")["dist_roads_m"].min().rename("dist_roads_m")
    )

    logger.info("Calculando distancia a poblados...")
    settlements_gdf = gpd.read_file(settlements_path).to_crs(grid_gdf.crs)
    joined_sett = gpd.sjoin_nearest(
        centroids_gdf,
        settlements_gdf[["geometry"]],
        how="left",
        distance_col="dist_settlements_m",
    )
    sett_dist = (
        joined_sett.groupby("cell_id")["dist_settlements_m"]
        .min()
        .rename("dist_settlements_m")
    )

    df = pd.DataFrame(
        {
            "cell_id": grid_gdf["cell_id"].values,
        }
    )
    df = df.join(roads_dist, on="cell_id")
    df = df.join(sett_dist, on="cell_id")

    df["dist_roads_km"] = df["dist_roads_m"] / 1_000.0
    df["dist_settlements_km"] = df["dist_settlements_m"] / 1_000.0
    df = df.drop(columns=["dist_roads_m", "dist_settlements_m"])

    logger.info(f"Distancias: {len(df):,} celdas procesadas")
    return df


# ---------------------------------------------------------------------------
# Áreas protegidas
# ---------------------------------------------------------------------------


def compute_protected_areas(
    grid_gdf: gpd.GeoDataFrame,
    protected_path: Path,
) -> pd.DataFrame:
    """Determina si cada celda intersecta un área protegida.

    Args:
        grid_gdf: GeoDataFrame de la grilla (EPSG:32616).
        protected_path: GeoPackage de áreas protegidas.

    Returns:
        DataFrame con columnas cell_id, is_protected_area (bool).

    Raises:
        FileNotFoundError: Si la capa no existe.
    """
    if not protected_path.exists():
        raise FileNotFoundError(f"Capa de áreas protegidas no encontrada: {protected_path}")

    logger.info("Calculando intersección con áreas protegidas...")
    protected_gdf = gpd.read_file(protected_path).to_crs(grid_gdf.crs)

    joined = gpd.sjoin(
        grid_gdf[["cell_id", "geometry"]],
        protected_gdf[["geometry"]],
        how="left",
        predicate="intersects",
    )
    protected_cells = set(joined.dropna(subset=["index_right"])["cell_id"].unique())

    df = pd.DataFrame(
        {
            "cell_id": grid_gdf["cell_id"].values,
            "is_protected_area": grid_gdf["cell_id"].isin(protected_cells),
        }
    )
    n_protected = df["is_protected_area"].sum()
    logger.info(
        f"Áreas protegidas: {n_protected:,} celdas ({100 * n_protected / len(df):.1f}%)"
    )
    return df


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------


def build_static_features(
    cfg: Settings | None = None,
    grid_path: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Compila todas las features estáticas por celda.

    Args:
        cfg: Configuración del proyecto.
        grid_path: Ruta al GeoPackage de la grilla. Por defecto data/interim/peten_grid.gpkg.
        output_path: Ruta de salida del parquet. Por defecto data/processed/static_features.parquet.

    Returns:
        DataFrame con una fila por cell_id y las columnas de features estáticas.

    Raises:
        FileNotFoundError: Si algún input requerido no existe.
    """
    cfg = cfg or default_settings
    grid_path = grid_path or (cfg.data_interim / "peten_grid.gpkg")
    output_path = output_path or (cfg.data_processed / "static_features.parquet")

    if not grid_path.exists():
        raise FileNotFoundError(
            f"Grilla no encontrada: {grid_path}. "
            "Ejecuta primero: uv run python -m src.data.build_grid"
        )

    logger.info(f"Cargando grilla desde {grid_path}...")
    grid_gdf = gpd.read_file(grid_path, layer="peten_grid")
    if grid_gdf.crs.to_epsg() != 32616:
        raise ValueError(f"CRS de la grilla debe ser EPSG:32616, recibido: {grid_gdf.crs}")
    logger.info(f"Grilla: {len(grid_gdf):,} celdas")

    topo_dir = cfg.data_interim / "topography"
    static_dir = cfg.data_interim / "static"

    # Topografía
    topo_df = compute_topo_features(grid_gdf, topo_dir)

    # Distancias
    dist_df = compute_distance_features(
        grid_gdf,
        roads_path=static_dir / "roads_peten.gpkg",
        settlements_path=static_dir / "settlements_peten.gpkg",
    )

    # Áreas protegidas
    prot_df = compute_protected_areas(
        grid_gdf,
        protected_path=static_dir / "protected_areas_peten.gpkg",
    )

    # Unir todo sobre cell_id
    result = topo_df.merge(dist_df, on="cell_id", how="left")
    result = result.merge(prot_df, on="cell_id", how="left")

    logger.info(f"Features estáticas: {result.shape[1] - 1} features x {len(result):,} celdas")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(output_path, index=False)
        logger.success(f"Guardado en {output_path}")

    return result


@app.command()
def main(
    grid_path: Annotated[
        Path,
        typer.Option("--grid-path", "-g", help="Ruta al GeoPackage de la grilla"),
    ] = Path("data/interim/peten_grid.gpkg"),
    output_path: Annotated[
        Path,
        typer.Option("--output-path", "-o", help="Ruta del parquet de salida"),
    ] = Path("data/processed/static_features.parquet"),
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Nivel de logging"),
    ] = "INFO",
) -> None:
    """Compila features estáticas por celda (topografía, distancias, áreas protegidas)."""
    logger.remove()
    logger.add(sys.stderr, level=log_level.upper())
    build_static_features(grid_path=grid_path, output_path=output_path)


if __name__ == "__main__":
    app()

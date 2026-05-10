"""Construye la grilla 1 km x 1 km del Petén en EPSG:32616.

Uso:
    uv run python -m src.data.build_grid
    uv run python -m src.data.build_grid --output-path data/interim/peten_grid.gpkg
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import geopandas as gpd
import numpy as np
import typer
from loguru import logger
from pyproj import Transformer
from shapely.geometry import box

from src.data.config import Settings
from src.data.config import settings as default_settings

app = typer.Typer(add_completion=False)


def build_grid(
    cfg: Settings | None = None,
    output_path: Path | None = None,
) -> gpd.GeoDataFrame:
    """Construye la grilla 1 km x 1 km del Petén.

    Args:
        cfg: Configuración del proyecto. Usa `settings` global si es None.
        output_path: Ruta de salida del GeoPackage. Si es None, no guarda.

    Returns:
        GeoDataFrame con columnas `cell_id`, `row`, `col`, `lon_center`,
        `lat_center`, `easting_m`, `northing_m` y geometría Polygon en
        EPSG:32616. Solo incluye celdas dentro del bbox del Petén.
    """
    cfg = cfg or default_settings
    res_m = cfg.grid_resolution_km * 1_000  # metros

    # Transformadores entre CRS
    to_utm = Transformer.from_crs(cfg.crs_geo, cfg.crs_work, always_xy=True)
    to_geo = Transformer.from_crs(cfg.crs_work, cfg.crs_geo, always_xy=True)

    lon_min, lat_min, lon_max, lat_max = cfg.peten_bbox

    # Convertir bbox a UTM 16N
    x_min, y_min = to_utm.transform(lon_min, lat_min)
    x_max, y_max = to_utm.transform(lon_max, lat_max)

    # Alinear a múltiplos de la resolución para celdas limpias
    x_min = np.floor(x_min / res_m) * res_m
    y_min = np.floor(y_min / res_m) * res_m
    x_max = np.ceil(x_max / res_m) * res_m
    y_max = np.ceil(y_max / res_m) * res_m

    xs = np.arange(x_min, x_max, res_m)
    ys = np.arange(y_max, y_min, -res_m)  # norte → sur

    n_cols = len(xs)
    n_rows = len(ys)
    logger.info(f"Grilla: {n_rows} filas x {n_cols} columnas = {n_rows * n_cols:,} celdas")

    geometries = []
    cell_ids = []
    rows_idx = []
    cols_idx = []
    eastings = []
    northings = []
    lon_centers = []
    lat_centers = []

    for r, y0 in enumerate(ys):
        for c, x0 in enumerate(xs):
            cx = x0 + res_m / 2
            cy = y0 - res_m / 2
            lon_c, lat_c = to_geo.transform(cx, cy)

            geometries.append(box(x0, y0 - res_m, x0 + res_m, y0))
            cell_ids.append(f"r{r:04d}_c{c:04d}")
            rows_idx.append(r)
            cols_idx.append(c)
            eastings.append(cx)
            northings.append(cy)
            lon_centers.append(lon_c)
            lat_centers.append(lat_c)

    gdf = gpd.GeoDataFrame(
        {
            "cell_id": cell_ids,
            "row": rows_idx,
            "col": cols_idx,
            "lon_center": lon_centers,
            "lat_center": lat_centers,
            "easting_m": eastings,
            "northing_m": northings,
        },
        geometry=geometries,
        crs=cfg.crs_work,
    )

    logger.info(f"GeoDataFrame creado: {len(gdf):,} celdas, CRS={gdf.crs}")
    _validate_grid(gdf)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(output_path, driver="GPKG", layer="peten_grid")
        logger.success(f"Grilla guardada en {output_path}  ({len(gdf):,} celdas)")

    return gdf


def _validate_grid(gdf: gpd.GeoDataFrame) -> None:
    """Verifica invariantes básicos de la grilla.

    Args:
        gdf: GeoDataFrame de la grilla a validar.

    Raises:
        ValueError: Si alguna validación falla.
    """
    if gdf.crs is None or gdf.crs.to_epsg() != 32616:
        raise ValueError(f"CRS incorrecto: {gdf.crs}. Esperado EPSG:32616")

    null_geoms = gdf.geometry.isna().sum()
    if null_geoms > 0:
        raise ValueError(f"{null_geoms} celdas con geometría nula")

    invalid_geoms = (~gdf.geometry.is_valid).sum()
    if invalid_geoms > 0:
        raise ValueError(f"{invalid_geoms} celdas con geometría inválida")

    if gdf["cell_id"].duplicated().any():
        raise ValueError("cell_id duplicado en la grilla")

    # Verificar que las áreas son consistentes (~1 km²)
    areas_km2 = gdf.geometry.area / 1e6
    if not (areas_km2.between(0.99, 1.01).all()):
        raise ValueError(
            f"Área de celdas fuera de rango: min={areas_km2.min():.4f}, max={areas_km2.max():.4f} km²"
        )

    logger.debug("Validación de grilla: OK")


@app.command()
def main(
    output_path: Annotated[
        Path,
        typer.Option("--output-path", "-o", help="Ruta de salida del GeoPackage"),
    ] = Path("data/interim/peten_grid.gpkg"),
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Nivel de logging"),
    ] = "INFO",
) -> None:
    """Construye y guarda la grilla 1 km x 1 km del Petén."""
    logger.remove()
    logger.add(sys.stderr, level=log_level.upper())

    build_grid(output_path=output_path)


if __name__ == "__main__":
    app()

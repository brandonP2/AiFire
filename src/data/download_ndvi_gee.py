"""Descarga NDVI de MODIS MOD13Q1 para el Petén usando Google Earth Engine.

Ventaja sobre descarga directa Earthdata: GEE recorta al bbox, reproyecta
y exporta directamente a Google Drive sin descargar los HDF globales (~400 MB c/u).

Flujo:
    1. Por cada composición 16 días (MOD13Q1) disponible en el rango de fechas,
       lanza un Export.image.toDrive en GEE.
    2. Monitorea el estado de los tasks hasta que completan.
    3. Descarga los GeoTIFF exportados desde Google Drive (o los descarga
       directamente vía getDownloadURL para tiles pequeños).

Genera:
    data/interim/ndvi/ndvi_YYYYDDD.tif  (EPSG:32616, 250m, valores [-1, 1])

Uso:
    uv run python -m src.data.download_ndvi_gee --year 2024
    uv run python -m src.data.download_ndvi_gee --start 2018 --end 2024
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Annotated

import ee
import httpx
import numpy as np
import rasterio
import rasterio.transform
import typer
from loguru import logger

from src.data.config import Settings
from src.data.config import settings as default_settings

app = typer.Typer(add_completion=False)

COLLECTION = "MODIS/061/MOD13Q1"
NDVI_BAND = "NDVI"
NDVI_SCALE = 0.0001
NDVI_NODATA = -3000
EXPORT_SCALE = 250  # metros — resolución nativa MOD13Q1
EXPORT_CRS = "EPSG:32616"
MAX_PIXELS = 1e9


def _init_gee(cfg: Settings) -> None:
    """Inicializa GEE con las credenciales del service account.

    Args:
        cfg: Configuración del proyecto con credenciales GEE.

    Raises:
        RuntimeError: Si las credenciales no están configuradas o son inválidas.
    """
    if not cfg.gee_ready:
        raise RuntimeError(
            "Credenciales GEE no configuradas. "
            "Verificar GEE_SERVICE_ACCOUNT y GEE_PRIVATE_KEY_PATH en .env.local"
        )
    credentials = ee.ServiceAccountCredentials(
        cfg.gee_service_account,
        str(cfg.gee_private_key_path),
    )
    ee.Initialize(credentials)


def _bbox_to_geometry(bbox: tuple[float, float, float, float]) -> ee.Geometry:
    """Convierte el bbox del Petén a geometría GEE.

    Args:
        bbox: (lon_min, lat_min, lon_max, lat_max) en WGS84.

    Returns:
        ee.Geometry.Rectangle en WGS84.
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    return ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])


def _download_chip_direct(
    image: ee.Image,
    roi: ee.Geometry,
    output_path: Path,
    scale: int = EXPORT_SCALE,
    crs: str = EXPORT_CRS,
) -> bool:
    """Descarga un chip de imagen GEE directamente vía getDownloadURL.

    Adecuado para el Petén (~35,854 km²) a 250m resolución.
    Límite práctico: ~100 MB por descarga directa.

    Args:
        image: ee.Image a descargar (ya recortada a ROI si es necesario).
        roi: Región de interés.
        output_path: Ruta de destino del GeoTIFF.
        scale: Resolución en metros.
        crs: CRS de destino.

    Returns:
        True si la descarga fue exitosa.
    """
    try:
        url = image.getDownloadURL(
            {
                "region": roi,
                "scale": scale,
                "crs": crs,
                "format": "GEO_TIFF",
                "bands": [NDVI_BAND],
            }
        )
    except ee.EEException as exc:
        logger.error(f"GEE getDownloadURL falló: {exc}")
        return False

    try:
        with httpx.stream("GET", url, timeout=300, follow_redirects=True) as resp:
            if resp.status_code != 200:
                logger.error(f"HTTP {resp.status_code} descargando chip")
                return False
            output_path.write_bytes(resp.read())
    except httpx.RequestError as exc:
        logger.error(f"Error de red descargando chip: {exc}")
        return False

    # Aplicar escala NDVI al GeoTIFF descargado
    _apply_ndvi_scale(output_path)
    return True


def _apply_ndvi_scale(tif_path: Path) -> None:
    """Convierte el NDVI crudo (entero) a float32 en [-1, 1] in-place.

    Args:
        tif_path: Ruta al GeoTIFF con valores NDVI en escala raw (x 0.0001).
    """
    with rasterio.open(tif_path) as src:
        raw = src.read(1).astype(np.float32)
        profile = src.profile.copy()

    ndvi = np.where(raw == NDVI_NODATA, np.nan, raw * NDVI_SCALE)
    ndvi = np.clip(ndvi, -1.0, 1.0)

    profile.update(dtype=np.float32, nodata=np.nan)
    with rasterio.open(tif_path, "w", **profile) as dst:
        dst.write(ndvi, 1)


def download_ndvi_year(
    year: int,
    bbox: tuple[float, float, float, float],
    output_dir: Path,
    cfg: Settings,
    scale: int = EXPORT_SCALE,
) -> list[Path]:
    """Descarga todas las composiciones NDVI de un año para el Petén.

    Args:
        year: Año a descargar.
        bbox: Bounding box del Petén en WGS84.
        output_dir: Directorio de salida para los GeoTIFF.
        cfg: Configuración con credenciales GEE.
        scale: Resolución en metros (default 250m).

    Returns:
        Lista de rutas a los GeoTIFF generados.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    roi = _bbox_to_geometry(bbox)

    col = (
        ee.ImageCollection(COLLECTION)
        .filterBounds(roi)
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .select(NDVI_BAND)
    )

    images = col.toList(col.size()).getInfo()
    logger.info(f"GEE MOD13Q1 {year}: {len(images)} composiciones disponibles")

    results: list[Path] = []
    for img_info in images:
        img_id = img_info["id"]
        date_str = img_id.split("/")[-1]  # e.g. "2024_001"
        # Normalizar: algunos IDs usan "_", otros "."
        date_clean = date_str.replace(".", "_").replace("-", "_")
        out_path = output_dir / f"ndvi_{date_clean}.tif"

        if out_path.exists():
            logger.debug(f"Ya existe: {out_path.name}")
            results.append(out_path)
            continue

        logger.info(f"Descargando {date_clean}...")
        image = ee.Image(img_id).select(NDVI_BAND)

        ok = _download_chip_direct(image, roi, out_path, scale=scale)
        if ok:
            size_mb = out_path.stat().st_size / 1e6
            logger.success(f"  ✓ {out_path.name}  ({size_mb:.1f} MB)")
            results.append(out_path)
        else:
            logger.error(f"  ✗ Falló: {date_clean}")

        time.sleep(0.5)  # evitar saturar la API GEE

    return results


def download_ndvi(
    start_year: int,
    end_year: int,
    bbox: tuple[float, float, float, float],
    output_dir: Path,
    cfg: Settings,
) -> dict[int, list[Path]]:
    """Descarga NDVI GEE para un rango de años.

    Args:
        start_year: Año de inicio.
        end_year: Año de fin (inclusive).
        bbox: Bounding box del Petén.
        output_dir: Directorio de salida.
        cfg: Configuración del proyecto.

    Returns:
        Dict {año: [rutas de GeoTIFF]}.
    """
    _init_gee(cfg)
    all_paths: dict[int, list[Path]] = {}

    for year in range(start_year, end_year + 1):
        paths = download_ndvi_year(year, bbox, output_dir, cfg)
        all_paths[year] = paths
        total = sum(len(v) for v in all_paths.values())
        logger.info(f"Año {year}: {len(paths)} tiles | Acumulado: {total}")

    return all_paths


@app.command()
def main(
    year: Annotated[int | None, typer.Option("--year", help="Año específico")] = None,
    start: Annotated[int, typer.Option("--start", help="Año de inicio")] = 2018,
    end: Annotated[int, typer.Option("--end", help="Año de fin")] = 2024,
    output_dir: Annotated[Path, typer.Option("--output-dir")] = Path("data/interim/ndvi"),
    scale: Annotated[int, typer.Option("--scale", help="Resolución en metros")] = EXPORT_SCALE,
    log_level: Annotated[str, typer.Option("--log-level")] = "INFO",
) -> None:
    """Descarga MODIS MOD13Q1 NDVI para el Petén vía Google Earth Engine."""
    logger.remove()
    logger.add(sys.stderr, level=log_level.upper())

    cfg = default_settings
    if not cfg.gee_ready:
        logger.error("GEE no configurado. Verificar GEE_SERVICE_ACCOUNT y GEE_PRIVATE_KEY_PATH")
        raise typer.Exit(1)

    _init_gee(cfg)

    years = [year] if year else range(start, end + 1)
    total = 0
    for yr in years:
        paths = download_ndvi_year(yr, cfg.peten_bbox, output_dir, cfg, scale=scale)
        total += len(paths)

    logger.success(f"Total: {total} GeoTIFF de NDVI descargados")


if __name__ == "__main__":
    app()

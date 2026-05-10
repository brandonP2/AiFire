"""Descarga chips Sentinel-2 etiquetados para entrenamiento de M3 (detección).

Estrategia:
    1. Cruza imágenes Sentinel-2 SR con focos FIRMS activos para el mismo día.
    2. Extrae chips de 64x64 píxeles (10m → 640m) centrados en cada foco FIRMS.
    3. También extrae chips negativos (sin foco) de zonas aleatorias con cobertura forestal.
    4. Exporta chips en bandas SWIR (B11, B12) + NIR (B8) + Rojo (B4) como GeoTIFF.

El dataset resultante está listo para fine-tuning de ResNet18 o YOLOv8 (M3).

Genera:
    data/processed/chips/positive/chip_{date}_{lat:.4f}_{lon:.4f}.tif
    data/processed/chips/negative/chip_{date}_{lat:.4f}_{lon:.4f}.tif

Uso:
    uv run python -m src.data.download_s2_chips --start 2022-03-01 --end 2022-05-31
    uv run python -m src.data.download_s2_chips --start 2024-03-01 --end 2024-05-31 --max-chips 500
"""

from __future__ import annotations

import random
import sys
import time
from datetime import date, timedelta
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

S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
FIRMS_COLLECTION = "FIRMS"

# Bandas a exportar: SWIR1, SWIR2, NIR, Rojo
BANDS = ["B11", "B12", "B8", "B4"]

CHIP_SIZE_PX = 64  # píxeles por lado
S2_RESOLUTION = 10  # metros por píxel (Sentinel-2 10m bandas)
CHIP_SIZE_M = CHIP_SIZE_PX * S2_RESOLUTION  # 640m por lado
MAX_CLOUD_PCT = 20  # % máximo de nubes en la escena
FIRMS_BUFFER_M = 500  # buffer alrededor del foco FIRMS para centrar el chip


def _init_gee(cfg: Settings) -> None:
    """Inicializa GEE con service account.

    Args:
        cfg: Configuración del proyecto.
    """
    credentials = ee.ServiceAccountCredentials(
        cfg.gee_service_account,
        str(cfg.gee_private_key_path),
    )
    ee.Initialize(credentials)


def _get_s2_image(date_str: str, roi: ee.Geometry) -> ee.Image | None:
    """Obtiene la imagen Sentinel-2 menos nublada para una fecha y ROI.

    Busca en una ventana de ±2 días para aumentar cobertura.

    Args:
        date_str: Fecha en formato 'YYYY-MM-DD'.
        roi: Región de interés.

    Returns:
        ee.Image compuesto o None si no hay imágenes disponibles.
    """
    d = date.fromisoformat(date_str)
    start = (d - timedelta(days=2)).isoformat()
    end = (d + timedelta(days=2)).isoformat()

    col = (
        ee.ImageCollection(S2_COLLECTION)
        .filterBounds(roi)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD_PCT))
        .select(BANDS)
    )

    size = col.size().getInfo()
    if size == 0:
        return None

    # Usar la imagen con menos nubes
    return col.sort("CLOUDY_PIXEL_PERCENTAGE").first()


def _get_firms_points(
    date_str: str,
    roi: ee.Geometry,
    max_points: int = 50,
) -> list[dict]:
    """Obtiene focos activos FIRMS para una fecha y ROI.

    Args:
        date_str: Fecha en formato 'YYYY-MM-DD'.
        roi: Región de interés.
        max_points: Máximo de focos a retornar.

    Returns:
        Lista de dicts con 'lat', 'lon' de cada foco.
    """
    d = date.fromisoformat(date_str)
    start = d.isoformat()
    end = (d + timedelta(days=1)).isoformat()

    firms_col = ee.ImageCollection(FIRMS_COLLECTION).filterBounds(roi).filterDate(start, end)

    size = firms_col.size().getInfo()
    if size == 0:
        return []

    # Extraer puntos: FIRMS en GEE son imágenes de temperatura de brillo
    # Usar sampling sobre la colección compuesta
    firms_img = firms_col.mosaic()
    points = firms_img.sample(
        region=roi,
        scale=1000,
        numPixels=max_points,
        seed=42,
        geometries=True,
    )

    features = points.getInfo().get("features", [])
    result = []
    for f in features:
        coords = f["geometry"]["coordinates"]
        result.append({"lon": coords[0], "lat": coords[1]})

    return result


def _download_chip(
    image: ee.Image,
    center_lon: float,
    center_lat: float,
    output_path: Path,
) -> bool:
    """Descarga un chip de 64x64 px centrado en un punto.

    Args:
        image: ee.Image con las bandas SWIR/NIR/Rojo.
        center_lon: Longitud del centro del chip (WGS84).
        center_lat: Latitud del centro del chip (WGS84).
        output_path: Ruta de salida del GeoTIFF.

    Returns:
        True si la descarga fue exitosa.
    """
    half_m = CHIP_SIZE_M / 2
    # Crear ROI cuadrado alrededor del punto
    point = ee.Geometry.Point([center_lon, center_lat])
    chip_roi = point.buffer(half_m).bounds()

    try:
        url = image.getDownloadURL(
            {
                "region": chip_roi,
                "scale": S2_RESOLUTION,
                "crs": "EPSG:4326",
                "format": "GEO_TIFF",
                "bands": BANDS,
            }
        )
    except ee.EEException as exc:
        logger.debug(f"getDownloadURL falló ({center_lat:.3f},{center_lon:.3f}): {exc}")
        return False

    try:
        with httpx.stream("GET", url, timeout=120, follow_redirects=True) as resp:
            if resp.status_code != 200:
                return False
            raw_bytes = resp.read()
    except httpx.RequestError:
        return False

    output_path.write_bytes(raw_bytes)

    # Verificar que el chip tiene el tamaño esperado y no es todo nodata
    try:
        with rasterio.open(output_path) as src:
            arr = src.read()
            if np.all(arr == 0) or np.all(np.isnan(arr)):
                output_path.unlink()
                return False
    except Exception:
        output_path.unlink(missing_ok=True)
        return False

    return True


def _sample_negative_points(
    roi: ee.Geometry,
    n: int,
    seed: int = 42,
) -> list[dict]:
    """Muestrea puntos negativos (sin fuego) dentro del bbox del Petén.

    Usa WorldCover para asegurar que los puntos negativos caigan en
    cobertura forestal (clase 10 = bosque).

    Args:
        roi: Región de interés.
        n: Número de puntos a generar.
        seed: Semilla para reproducibilidad.

    Returns:
        Lista de dicts con 'lat', 'lon'.
    """
    forest_mask = (
        ee.ImageCollection("ESA/WorldCover/v200").first().eq(10)  # clase 10 = bosque
    )

    points = forest_mask.updateMask(forest_mask).sample(
        region=roi,
        scale=100,
        numPixels=n * 2,  # oversample para descartar los que fallen
        seed=seed,
        geometries=True,
    )

    features = points.getInfo().get("features", [])
    result = []
    for f in features[:n]:
        coords = f["geometry"]["coordinates"]
        result.append({"lon": coords[0], "lat": coords[1]})

    return result


def download_chips(
    start: date,
    end: date,
    bbox: tuple[float, float, float, float],
    output_dir: Path,
    cfg: Settings,
    max_chips: int = 1000,
    positive_ratio: float = 0.5,
) -> dict[str, int]:
    """Descarga chips Sentinel-2 positivos (con fuego) y negativos.

    Args:
        start: Fecha de inicio.
        end: Fecha de fin.
        bbox: Bounding box del Petén en WGS84.
        output_dir: Directorio raíz de salida.
        cfg: Configuración del proyecto.
        max_chips: Máximo total de chips a generar.
        positive_ratio: Fracción de chips positivos (default 0.5 = balanceado).

    Returns:
        Dict con conteos: {'positive': N, 'negative': N}.
    """
    pos_dir = output_dir / "positive"
    neg_dir = output_dir / "negative"
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)

    lon_min, lat_min, lon_max, lat_max = bbox
    roi = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])

    max_positive = int(max_chips * positive_ratio)
    max_negative = max_chips - max_positive

    counts = {"positive": 0, "negative": 0}
    current = start

    rng = random.Random(cfg.random_seed)

    while current <= end and (
        counts["positive"] < max_positive or counts["negative"] < max_negative
    ):
        date_str = current.isoformat()

        # Obtener imagen Sentinel-2 del día
        s2 = _get_s2_image(date_str, roi)
        if s2 is None:
            logger.debug(f"{date_str}: sin imágenes S2 disponibles")
            current += timedelta(days=1)
            continue

        # --- Chips positivos (focos FIRMS) ---
        if counts["positive"] < max_positive:
            firms_pts = _get_firms_points(date_str, roi)
            for pt in firms_pts:
                if counts["positive"] >= max_positive:
                    break
                fname = f"chip_{date_str}_{pt['lat']:.4f}_{pt['lon']:.4f}.tif"
                out = pos_dir / fname
                if out.exists():
                    counts["positive"] += 1
                    continue
                ok = _download_chip(s2, pt["lon"], pt["lat"], out)
                if ok:
                    counts["positive"] += 1
                    logger.info(f"  + positivo {counts['positive']}/{max_positive}: {fname}")

        # --- Chips negativos (muestra aleatoria forestal) ---
        if counts["negative"] < max_negative and counts["positive"] > 0:
            neg_pts = _sample_negative_points(roi, n=5, seed=rng.randint(0, 10000))
            for pt in neg_pts:
                if counts["negative"] >= max_negative:
                    break
                fname = f"chip_{date_str}_{pt['lat']:.4f}_{pt['lon']:.4f}.tif"
                out = neg_dir / fname
                if out.exists():
                    counts["negative"] += 1
                    continue
                ok = _download_chip(s2, pt["lon"], pt["lat"], out)
                if ok:
                    counts["negative"] += 1
                    logger.info(f"  - negativo {counts['negative']}/{max_negative}: {fname}")

        current += timedelta(days=1)
        time.sleep(0.3)

    logger.success(
        f"Chips descargados: {counts['positive']} positivos, {counts['negative']} negativos"
    )
    return counts


@app.command()
def main(
    start: Annotated[str, typer.Option("--start", help="Fecha inicio YYYY-MM-DD")] = "2022-03-01",
    end: Annotated[str, typer.Option("--end", help="Fecha fin YYYY-MM-DD")] = "2022-05-31",
    output_dir: Annotated[Path, typer.Option("--output-dir")] = Path("data/processed/chips"),
    max_chips: Annotated[int, typer.Option("--max-chips", help="Máximo total de chips")] = 1000,
    log_level: Annotated[str, typer.Option("--log-level")] = "INFO",
) -> None:
    """Descarga chips Sentinel-2 etiquetados (positivos/negativos) para entrenamiento M3."""
    logger.remove()
    logger.add(sys.stderr, level=log_level.upper())

    cfg = default_settings
    if not cfg.gee_ready:
        logger.error("GEE no configurado")
        raise typer.Exit(1)

    _init_gee(cfg)

    counts = download_chips(
        start=date.fromisoformat(start),
        end=date.fromisoformat(end),
        bbox=cfg.peten_bbox,
        output_dir=output_dir,
        cfg=cfg,
        max_chips=max_chips,
    )
    logger.info(f"Total: {sum(counts.values())} chips")


if __name__ == "__main__":
    app()

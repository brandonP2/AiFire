"""Descarga topografía SRTM 30m para el Petén desde AWS terrain tiles.

Sin autenticación requerida. Tiles en formato HGT (.hgt.gz) desde:
https://s3.amazonaws.com/elevation-tiles-prod/skadi/

Genera:
    data/interim/topography/srtm_peten.tif     — elevación en metros
    data/interim/topography/slope_peten.tif    — pendiente en grados
    data/interim/topography/aspect_peten.tif   — aspecto en grados (0=norte)

Uso:
    uv run python -m src.data.download_srtm
    uv run python -m src.data.download_srtm --output-dir data/interim/topography
"""

from __future__ import annotations

import gzip
import math
import struct
import sys
import tempfile
from pathlib import Path
from typing import Annotated

import httpx
import numpy as np
import rasterio
import rasterio.crs
import rasterio.merge
import rasterio.transform
import rasterio.warp
import typer
from loguru import logger
from rasterio.enums import Resampling

from src.data.config import settings as default_settings

app = typer.Typer(add_completion=False)

AWS_BASE = "https://s3.amazonaws.com/elevation-tiles-prod/skadi"
SRTM_PIXELS = 3601  # píxeles por grado en SRTM 1 arc-second (~30m)
NODATA = -32768


def _tile_url(lat: int, lon: int) -> str:
    """Construye la URL del tile AWS para un par lat/lon enteros.

    Args:
        lat: Latitud entera del tile (e.g., 16 para N16).
        lon: Longitud entera del tile (e.g., -91 para W091).
    """
    lat_prefix = "N" if lat >= 0 else "S"
    lon_prefix = "E" if lon >= 0 else "W"
    lat_abs = abs(lat)
    lon_abs = abs(lon)
    tile_name = f"{lat_prefix}{lat_abs:02d}{lon_prefix}{lon_abs:03d}"
    prefix = f"{lat_prefix}{lat_abs:02d}"
    return f"{AWS_BASE}/{prefix}/{tile_name}.hgt.gz", tile_name


def _download_hgt(url: str, tile_name: str, cache_dir: Path) -> Path | None:
    """Descarga y descomprime un tile HGT desde AWS.

    Args:
        url: URL del archivo .hgt.gz.
        tile_name: Nombre del tile (e.g., N16W091).
        cache_dir: Directorio de caché para tiles crudos.

    Returns:
        Ruta al archivo .hgt descomprimido, o None si falló.
    """
    hgt_path = cache_dir / f"{tile_name}.hgt"
    if hgt_path.exists():
        logger.debug(f"Tile en caché: {tile_name}")
        return hgt_path

    gz_path = cache_dir / f"{tile_name}.hgt.gz"
    logger.info(f"Descargando {tile_name}...")

    try:
        with httpx.stream("GET", url, timeout=120, follow_redirects=True) as resp:
            if resp.status_code == 404:
                logger.warning(f"Tile no encontrado: {tile_name} (zona oceánica o sin datos)")
                return None
            if resp.status_code != 200:
                logger.error(f"HTTP {resp.status_code} para {tile_name}")
                return None
            gz_path.write_bytes(resp.read())
    except httpx.RequestError as exc:
        logger.error(f"Error descargando {tile_name}: {exc}")
        return None

    # Descomprimir
    with gzip.open(gz_path, "rb") as f_in:
        hgt_path.write_bytes(f_in.read())
    gz_path.unlink()
    logger.debug(f"Descomprimido: {hgt_path.name}  ({hgt_path.stat().st_size / 1e6:.1f} MB)")
    return hgt_path


def _hgt_to_tif(hgt_path: Path, lat: int, lon: int, out_path: Path) -> Path:
    """Convierte un archivo HGT binario a GeoTIFF con CRS WGS84.

    Args:
        hgt_path: Ruta al archivo .hgt.
        lat: Latitud entera del tile.
        lon: Longitud entera del tile.
        out_path: Ruta de salida del GeoTIFF.

    Returns:
        Ruta al GeoTIFF creado.
    """
    data_bytes = hgt_path.read_bytes()
    n = SRTM_PIXELS
    expected = n * n * 2  # int16, big-endian
    values = struct.unpack(f">{n * n}h", data_bytes[:expected])
    arr = np.array(values, dtype=np.int16).reshape(n, n)

    # Transform: el tile cubre 1°×1° empezando desde la esquina NW
    transform = rasterio.transform.from_bounds(
        west=lon, south=lat, east=lon + 1, north=lat + 1, width=n, height=n
    )
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=n,
        width=n,
        count=1,
        dtype=np.int16,
        crs=rasterio.crs.CRS.from_epsg(4326),
        transform=transform,
        nodata=NODATA,
    ) as dst:
        dst.write(arr, 1)

    return out_path


def _compute_slope_aspect(elevation: np.ndarray, cell_size_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Calcula pendiente y aspecto a partir de un raster de elevación.

    Args:
        elevation: Array 2D de elevación en metros.
        cell_size_m: Tamaño de celda en metros.

    Returns:
        Tupla (slope_deg, aspect_deg). Aspecto en grados desde el norte,
        en sentido horario. -1 donde la pendiente es 0 (terreno plano).
    """
    # Gradientes usando diferencias centrales
    dy, dx = np.gradient(elevation.astype(float), cell_size_m)

    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_deg = np.degrees(slope_rad)

    # Aspecto: 0 = norte, 90 = este, etc.
    aspect_rad = np.arctan2(-dx, dy)  # -dx porque x aumenta hacia el este
    aspect_deg = np.degrees(aspect_rad) % 360

    # Terreno plano → aspecto -1
    flat_mask = slope_deg < 0.01
    aspect_deg = np.where(flat_mask, -1.0, aspect_deg)

    return slope_deg.astype(np.float32), aspect_deg.astype(np.float32)


def download_srtm(
    bbox: tuple[float, float, float, float],
    output_dir: Path,
    target_crs: str = "EPSG:32616",
    target_resolution_m: float = 30.0,
) -> dict[str, Path]:
    """Descarga y procesa tiles SRTM para el bbox dado.

    Args:
        bbox: Bounding box (lon_min, lat_min, lon_max, lat_max) en WGS84.
        output_dir: Directorio de salida para los rasters procesados.
        target_crs: CRS de destino para reproyección.
        target_resolution_m: Resolución de salida en metros.

    Returns:
        Dict con rutas: {'elevation': ..., 'slope': ..., 'aspect': ...}
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "tiles_raw"
    cache_dir.mkdir(exist_ok=True)

    lon_min, lat_min, lon_max, lat_max = bbox
    lat_tiles = range(math.floor(lat_min), math.ceil(lat_max))
    lon_tiles = range(math.floor(lon_min), math.ceil(lon_max))

    logger.info(f"Tiles necesarios: lat {list(lat_tiles)}, lon {list(lon_tiles)}")

    # 1. Descargar y convertir tiles a GeoTIFF WGS84
    tile_tifs: list[Path] = []
    for lat in lat_tiles:
        for lon in lon_tiles:
            url, tile_name = _tile_url(lat, lon)
            hgt_path = _download_hgt(url, tile_name, cache_dir)
            if hgt_path is None:
                continue
            tif_path = cache_dir / f"{tile_name}.tif"
            if not tif_path.exists():
                _hgt_to_tif(hgt_path, lat, lon, tif_path)
            tile_tifs.append(tif_path)

    if not tile_tifs:
        raise RuntimeError("No se descargó ningún tile SRTM")

    logger.info(f"{len(tile_tifs)} tiles descargados — creando mosaico...")

    # 2. Mosaico de tiles en WGS84
    mosaic_path = cache_dir / "srtm_mosaic_wgs84.tif"
    if not mosaic_path.exists():
        datasets = [rasterio.open(p) for p in tile_tifs]
        mosaic, out_transform = rasterio.merge.merge(datasets, nodata=NODATA)
        for ds in datasets:
            ds.close()

        meta = datasets[0].meta.copy()
        meta.update(
            driver="GTiff",
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            transform=out_transform,
            nodata=NODATA,
            compress="lzw",
        )
        with rasterio.open(mosaic_path, "w", **meta) as dst:
            dst.write(mosaic)

    # 3. Reproyectar a CRS de trabajo y recortar al bbox del Petén
    elev_path = output_dir / "srtm_peten.tif"
    if not elev_path.exists():
        logger.info(f"Reproyectando a {target_crs}...")
        with rasterio.open(mosaic_path) as src:
            target_transform, target_width, target_height = rasterio.warp.calculate_default_transform(
                src.crs,
                target_crs,
                src.width,
                src.height,
                *src.bounds,
                resolution=target_resolution_m,
            )
            kwargs = src.meta.copy()
            kwargs.update(
                crs=target_crs,
                transform=target_transform,
                width=target_width,
                height=target_height,
                dtype=np.float32,
                nodata=float("nan"),
                compress="lzw",
            )
            with rasterio.open(elev_path, "w", **kwargs) as dst:
                rasterio.warp.reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=target_transform,
                    dst_crs=target_crs,
                    resampling=Resampling.bilinear,
                    src_nodata=NODATA,
                    dst_nodata=float("nan"),
                )

    logger.success(f"Elevación: {elev_path}")

    # 4. Calcular pendiente y aspecto
    slope_path = output_dir / "slope_peten.tif"
    aspect_path = output_dir / "aspect_peten.tif"

    if not slope_path.exists() or not aspect_path.exists():
        logger.info("Calculando pendiente y aspecto...")
        with rasterio.open(elev_path) as src:
            elevation = src.read(1)
            meta = src.meta.copy()
            # Resolución en metros (UTM, unidades métricas)
            cell_size = abs(src.transform.a)

        elevation = np.where(np.isnan(elevation), 0.0, elevation)
        slope, aspect = _compute_slope_aspect(elevation, cell_size)

        for path, arr, name in [(slope_path, slope, "pendiente"), (aspect_path, aspect, "aspecto")]:
            m = meta.copy()
            m.update(dtype=np.float32, nodata=float("nan"))
            with rasterio.open(path, "w", **m) as dst:
                dst.write(arr[np.newaxis, :, :])
            logger.success(f"{name.capitalize()}: {path}")

    return {"elevation": elev_path, "slope": slope_path, "aspect": aspect_path}


@app.command()
def main(
    output_dir: Annotated[Path, typer.Option("--output-dir")] = Path("data/interim/topography"),
    log_level: Annotated[str, typer.Option("--log-level")] = "INFO",
) -> None:
    """Descarga SRTM 30m para el Petén y genera rasters de pendiente y aspecto."""
    logger.remove()
    logger.add(sys.stderr, level=log_level.upper())

    cfg = default_settings
    result = download_srtm(cfg.peten_bbox, output_dir, cfg.crs_work)
    for key, path in result.items():
        logger.info(f"{key}: {path}")


if __name__ == "__main__":
    app()

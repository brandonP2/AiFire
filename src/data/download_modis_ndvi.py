"""Descarga y procesa NDVI de MODIS MOD13Q1 (250m, composición 16 días).

Usa NASA Earthdata (credenciales en .env.local) vía CMR para buscar granulas
y httpx con autenticación para descargar los HDF.
La estrategia de búsqueda usa bbox del Petén en lugar de ID de tile.

Genera:
    data/raw/ndvi/                  — HDF crudos por año
    data/interim/ndvi/              — GeoTIFF reproyectados a EPSG:32616

Uso:
    uv run python -m src.data.download_modis_ndvi --year 2024
    uv run python -m src.data.download_modis_ndvi --start 2018 --end 2024
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Annotated

import httpx
import numpy as np
import rasterio
import rasterio.crs
import rasterio.warp
import typer
from loguru import logger
from rasterio.enums import Resampling

from src.data.config import Settings
from src.data.config import settings as default_settings

app = typer.Typer(add_completion=False)

CMR_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
EARTHDATA_URS = "https://urs.earthdata.nasa.gov"
COLLECTION_SHORT_NAME = "MOD13Q1"
COLLECTION_VERSION = "061"

# Nombre de la subdataset NDVI dentro del HDF MOD13Q1
# Formato: HDF4_EOS:EOS_GRID:"file.hdf":MODIS_Grid_16DAY_250m_500m_VI:250m 16 days NDVI
NDVI_SUBDATASET = "250m 16 days NDVI"
NDVI_SCALE = 0.0001  # factor de escala para convertir a [-1, 1]
NDVI_NODATA_RAW = -3000


def _earthdata_session(user: str, password: str) -> httpx.Client:
    """Crea un cliente httpx autenticado con Earthdata URS.

    La autenticación de Earthdata usa cookies de sesión — el cliente
    sigue redirects automáticamente y mantiene las cookies.

    Args:
        user: Usuario de NASA Earthdata.
        password: Contraseña de NASA Earthdata.

    Returns:
        Cliente httpx autenticado.
    """
    client = httpx.Client(
        auth=(user, password),
        follow_redirects=True,
        timeout=120,
    )
    return client


def _search_granules(
    bbox: tuple[float, float, float, float],
    year: int,
) -> list[dict]:
    """Busca granulas MOD13Q1 en CMR para el bbox y año dado.

    Args:
        bbox: Bounding box (lon_min, lat_min, lon_max, lat_max) en WGS84.
        year: Año de búsqueda.

    Returns:
        Lista de metadatos de granulas con sus URLs de descarga.
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    params = {
        "short_name": COLLECTION_SHORT_NAME,
        "version": COLLECTION_VERSION,
        "temporal[]": f"{year}-01-01T00:00:00Z,{year}-12-31T23:59:59Z",
        "bounding_box": f"{lon_min},{lat_min},{lon_max},{lat_max}",
        "page_size": 50,
    }

    try:
        resp = httpx.get(CMR_URL, params=params, timeout=30)
    except httpx.RequestError as exc:
        logger.error(f"CMR error: {exc}")
        return []

    if resp.status_code != 200:
        logger.error(f"CMR HTTP {resp.status_code}: {resp.text[:200]}")
        return []

    entries = resp.json().get("feed", {}).get("entry", [])
    logger.info(f"CMR: {len(entries)} granulas para {year}")

    granules = []
    for entry in entries:
        links = entry.get("links", [])
        hdf_links = [
            lk["href"]
            for lk in links
            if lk.get("href", "").endswith(".hdf")
            and lk.get("rel") == "http://esipfed.org/ns/fedsearch/1.1/data#"
        ]
        if hdf_links:
            granules.append({
                "title": entry.get("title", "N/A"),
                "time_start": entry.get("time_start", "")[:10],
                "download_url": hdf_links[0],
            })

    return granules


def _download_hdf(
    url: str,
    output_path: Path,
    client: httpx.Client,
    max_retries: int = 3,
) -> bool:
    """Descarga un archivo HDF de Earthdata con autenticación.

    Args:
        url: URL del archivo HDF.
        output_path: Ruta de destino.
        client: Cliente httpx autenticado.
        max_retries: Reintentos ante fallos.

    Returns:
        True si la descarga fue exitosa.
    """
    if output_path.exists():
        logger.debug(f"Ya existe: {output_path.name}")
        return True

    for attempt in range(1, max_retries + 1):
        try:
            with client.stream("GET", url) as resp:
                if resp.status_code != 200:
                    logger.error(f"HTTP {resp.status_code} para {url}")
                    return False
                output_path.write_bytes(resp.read())
            size_mb = output_path.stat().st_size / 1e6
            logger.debug(f"Descargado: {output_path.name}  ({size_mb:.1f} MB)")
            return True
        except httpx.RequestError as exc:
            logger.warning(f"Red (intento {attempt}/{max_retries}): {exc}")
            if attempt < max_retries:
                time.sleep(2**attempt)
                output_path.unlink(missing_ok=True)

    return False


def _hdf_to_geotiff(
    hdf_path: Path,
    output_path: Path,
    target_crs: str = "EPSG:32616",
) -> bool:
    """Extrae la banda NDVI del HDF y la guarda como GeoTIFF en el CRS de trabajo.

    Args:
        hdf_path: Ruta al archivo HDF.
        output_path: Ruta del GeoTIFF de salida.
        target_crs: CRS de destino.

    Returns:
        True si el procesamiento fue exitoso.
    """
    if output_path.exists():
        return True

    # Buscar la subdataset NDVI
    subdataset_path = None
    try:
        with rasterio.open(str(hdf_path)) as src:
            for subds_name in src.subdatasets:
                if NDVI_SUBDATASET in subds_name:
                    subdataset_path = subds_name
                    break
    except Exception as exc:
        logger.error(f"Error abriendo HDF {hdf_path.name}: {exc}")
        return False

    if subdataset_path is None:
        logger.error(f"Subdataset NDVI no encontrado en {hdf_path.name}")
        return False

    try:
        with rasterio.open(subdataset_path) as src:
            raw = src.read(1)
            src_crs = src.crs
            src_transform = src.transform
            src_nodata = src.nodata or NDVI_NODATA_RAW

        # Aplicar escala y máscara de nodata
        ndvi = np.where(raw == src_nodata, np.nan, raw * NDVI_SCALE).astype(np.float32)

        # Reproyectar a CRS de trabajo
        target_transform, width, height = rasterio.warp.calculate_default_transform(
            src_crs, target_crs, raw.shape[1], raw.shape[0], transform=src_transform
        )
        ndvi_reproj = np.full((height, width), np.nan, dtype=np.float32)
        rasterio.warp.reproject(
            source=ndvi,
            destination=ndvi_reproj,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype=np.float32,
            crs=target_crs,
            transform=target_transform,
            nodata=np.nan,
            compress="lzw",
        ) as dst:
            dst.write(ndvi_reproj, 1)

    except Exception as exc:
        logger.error(f"Error procesando {hdf_path.name}: {exc}")
        output_path.unlink(missing_ok=True)
        return False

    return True


def download_modis_ndvi_year(
    year: int,
    bbox: tuple[float, float, float, float],
    earthdata_user: str,
    earthdata_pass: str,
    raw_dir: Path,
    interim_dir: Path,
    target_crs: str = "EPSG:32616",
) -> list[Path]:
    """Descarga y procesa todas las granulas MOD13Q1 de un año.

    Args:
        year: Año a procesar.
        bbox: Bounding box del Petén en WGS84.
        earthdata_user: Usuario NASA Earthdata.
        earthdata_pass: Contraseña NASA Earthdata.
        raw_dir: Directorio para HDF crudos.
        interim_dir: Directorio para GeoTIFF procesados.
        target_crs: CRS de destino.

    Returns:
        Lista de rutas a los GeoTIFF generados.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)

    granules = _search_granules(bbox, year)
    if not granules:
        logger.warning(f"Sin granulas para {year}")
        return []

    logger.info(f"Procesando {len(granules)} granulas de {year}...")
    client = _earthdata_session(earthdata_user, earthdata_pass)
    results: list[Path] = []

    for gran in granules:
        title = gran["title"]
        date_str = gran["time_start"].replace("-", "")
        hdf_path = raw_dir / f"{title}.hdf"
        tif_path = interim_dir / f"ndvi_{date_str}.tif"

        if tif_path.exists():
            results.append(tif_path)
            continue

        ok = _download_hdf(gran["download_url"], hdf_path, client)
        if not ok:
            logger.error(f"Descarga fallida: {title}")
            continue

        ok = _hdf_to_geotiff(hdf_path, tif_path, target_crs)
        if ok:
            results.append(tif_path)
            logger.success(f"NDVI procesado: {tif_path.name}")
        else:
            logger.error(f"Procesamiento fallido: {title}")

        time.sleep(1)

    client.close()
    return results


@app.command()
def main(
    year: Annotated[int | None, typer.Option("--year", help="Año específico")] = None,
    start: Annotated[int, typer.Option("--start", help="Año de inicio")] = 2018,
    end: Annotated[int, typer.Option("--end", help="Año de fin")] = 2024,
    raw_dir: Annotated[Path, typer.Option("--raw-dir")] = Path("data/raw/ndvi"),
    interim_dir: Annotated[Path, typer.Option("--interim-dir")] = Path("data/interim/ndvi"),
    log_level: Annotated[str, typer.Option("--log-level")] = "INFO",
) -> None:
    """Descarga y procesa MODIS MOD13Q1 NDVI para el Petén."""
    logger.remove()
    logger.add(sys.stderr, level=log_level.upper())

    cfg = default_settings
    if not cfg.earthdata_user or not cfg.earthdata_pass:
        logger.error("EARTHDATA_USER / EARTHDATA_PASS no configuradas")
        raise typer.Exit(1)

    years = [year] if year else range(start, end + 1)
    total: list[Path] = []

    for yr in years:
        paths = download_modis_ndvi_year(
            year=yr,
            bbox=cfg.peten_bbox,
            earthdata_user=cfg.earthdata_user,
            earthdata_pass=cfg.earthdata_pass,
            raw_dir=raw_dir,
            interim_dir=interim_dir,
            target_crs=cfg.crs_work,
        )
        total.extend(paths)
        logger.info(f"Año {yr}: {len(paths)} tiles generados")

    logger.success(f"Total: {len(total)} GeoTIFF de NDVI generados")


if __name__ == "__main__":
    app()

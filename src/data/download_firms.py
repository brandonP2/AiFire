"""Descarga focos activos de NASA FIRMS para el Petén.

Fuentes soportadas: VIIRS Suomi-NPP, VIIRS NOAA-20, MODIS Terra+Aqua.
API: https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{source}/{bbox}/{days}/{date}

Uso:
    uv run python -m src.data.download_firms --start 2018-01-01 --end 2024-12-31
    uv run python -m src.data.download_firms --start 2024-01-01 --end 2024-12-31 --source VIIRS_SNPP_NRT
"""

from __future__ import annotations

import sys
import time
from datetime import date, timedelta
from io import StringIO
from pathlib import Path
from typing import Annotated

import httpx
import pandas as pd
import typer
from loguru import logger

from src.data.config import Settings
from src.data.config import settings as default_settings

app = typer.Typer(add_completion=False)

BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
CHUNK_DAYS = 5  # máx 5 días por request en la API FIRMS
SLEEP_BETWEEN_CHUNKS = 2.0  # segundos — respetar rate limits

# MODIS usa 'brightness'; normalizar al nombre de VIIRS para schema unificado
_COLUMN_ALIASES: dict[str, str] = {"brightness": "bright_ti4"}

# Productos de archivo histórico (SP = Science Product, datos 2012-presente)
# y NRT para datos recientes (<90 días). Usar SP para descarga histórica.
SOURCES = (
    "VIIRS_SNPP_NRT",
    "VIIRS_NOAA20_NRT",
    "MODIS_NRT",
    "VIIRS_SNPP_SP",
    "VIIRS_NOAA20_SP",
    "MODIS_SP",
)

# Columnas mínimas requeridas (tras normalización)
_REQUIRED_COLS = {"latitude", "longitude", "acq_date", "acq_time", "confidence"}


def _bbox_str(bbox: tuple[float, float, float, float]) -> str:
    lon_min, lat_min, lon_max, lat_max = bbox
    return f"{lon_min},{lat_min},{lon_max},{lat_max}"


def _fetch_chunk(
    map_key: str,
    source: str,
    bbox_s: str,
    start_date: date,
    days: int,
    max_retries: int = 3,
) -> pd.DataFrame | None:
    """Descarga un chunk de hasta 10 días de FIRMS.

    Args:
        map_key: NASA FIRMS MAP key.
        source: Nombre del producto FIRMS.
        bbox_s: Bounding box como string CSV.
        start_date: Fecha de inicio del chunk.
        days: Número de días (1-10).
        max_retries: Reintentos ante fallos de red.

    Returns:
        DataFrame con los focos del chunk, DataFrame vacío si no hay focos,
        o None si el request falló definitivamente.
    """
    url = f"{BASE_URL}/{map_key}/{source}/{bbox_s}/{days}/{start_date.isoformat()}"

    for attempt in range(1, max_retries + 1):
        try:
            resp = httpx.get(url, timeout=60, follow_redirects=True)
        except httpx.RequestError as exc:
            logger.warning(f"Red (intento {attempt}/{max_retries}): {exc}")
            if attempt < max_retries:
                time.sleep(2**attempt)
            continue

        if resp.status_code == 429:
            logger.warning("Rate limit (429) — esperando 60s")
            time.sleep(60)
            continue

        if resp.status_code != 200:
            logger.error(f"HTTP {resp.status_code} para {start_date}: {resp.text[:200]}")
            return None

        text = resp.text.strip()
        if not text:
            return pd.DataFrame()

        lines = text.splitlines()
        if len(lines) <= 1:  # solo header, sin focos
            return pd.DataFrame()

        try:
            df = pd.read_csv(StringIO(text), low_memory=False)
        except Exception as exc:
            logger.error(f"Error CSV ({start_date}): {exc}")
            return None

        df = df.rename(columns=_COLUMN_ALIASES)
        df["source"] = source

        missing = _REQUIRED_COLS - set(df.columns)
        if missing:
            logger.warning(f"Columnas faltantes en {source}: {missing}")

        return df

    return None


def download_firms(
    start: date,
    end: date,
    source: str,
    map_key: str,
    bbox: tuple[float, float, float, float],
    output_dir: Path,
    chunk_days: int = CHUNK_DAYS,
    sleep_s: float = SLEEP_BETWEEN_CHUNKS,
) -> Path | None:
    """Descarga focos activos FIRMS para un rango de fechas y los guarda en CSV.

    Args:
        start: Fecha de inicio (inclusive).
        end: Fecha de fin (inclusive).
        source: Producto FIRMS (ver SOURCES).
        map_key: NASA FIRMS MAP key.
        bbox: Bounding box (lon_min, lat_min, lon_max, lat_max) en WGS84.
        output_dir: Directorio de salida.
        chunk_days: Días por chunk (máximo 10).
        sleep_s: Segundos de espera entre chunks.

    Returns:
        Ruta al CSV resultante, o None si no hubo focos.
    """
    if source not in SOURCES:
        raise ValueError(f"Fuente inválida: {source!r}. Opciones: {SOURCES}")

    output_dir.mkdir(parents=True, exist_ok=True)
    fname = f"firms_{source}_{start.year}-{start.month:02d}_to_{end.year}-{end.month:02d}.csv"
    output_path = output_dir / fname

    if output_path.exists():
        logger.info(f"Ya existe: {output_path} — saltando descarga")
        return output_path

    bbox_s = _bbox_str(bbox)
    chunks: list[pd.DataFrame] = []
    current = start
    total_days = (end - start).days + 1
    done_days = 0

    logger.info(f"FIRMS {source}: {start} → {end}  ({total_days} días, chunks de {chunk_days}d)")

    while current <= end:
        days = min(chunk_days, (end - current).days + 1)
        df = _fetch_chunk(map_key, source, bbox_s, current, days)

        if df is None:
            logger.error(f"Chunk {current} falló — abortando")
            return None
        if not df.empty:
            chunks.append(df)

        done_days += days
        current += timedelta(days=days)
        logger.debug(f"  {done_days}/{total_days}d — acumulados: {sum(len(c) for c in chunks):,} focos")
        time.sleep(sleep_s)

    if not chunks:
        logger.warning(f"Sin focos para {source} en {start}–{end} (normal fuera de temporada)")
        pd.DataFrame(columns=list(_REQUIRED_COLS) + ["source"]).to_csv(output_path, index=False)
        return output_path

    combined = (
        pd.concat(chunks, ignore_index=True)
        .drop_duplicates(subset=["latitude", "longitude", "acq_date", "acq_time"])
        .sort_values(["acq_date", "acq_time"])
    )
    combined.to_csv(output_path, index=False)
    logger.success(f"Guardado: {output_path}  ({len(combined):,} focos únicos)")
    return output_path


@app.command()
def main(
    start: Annotated[str, typer.Option("--start", help="Fecha inicio YYYY-MM-DD")] = "2024-01-01",
    end: Annotated[str, typer.Option("--end", help="Fecha fin YYYY-MM-DD")] = "2024-12-31",
    source: Annotated[str, typer.Option("--source", help="Fuente FIRMS (SP para histórico, NRT para reciente)")] = "VIIRS_SNPP_SP",
    all_sources: Annotated[bool, typer.Option("--all-sources", help="Descargar todas las fuentes")] = False,
    output_dir: Annotated[Path, typer.Option("--output-dir")] = Path("data/raw/firms"),
    log_level: Annotated[str, typer.Option("--log-level")] = "INFO",
) -> None:
    """Descarga focos activos NASA FIRMS para el Petén."""
    logger.remove()
    logger.add(sys.stderr, level=log_level.upper())

    cfg = default_settings
    if not cfg.firms_map_key:
        logger.error("FIRMS_MAP_KEY no configurada en .env.local")
        raise typer.Exit(1)

    start_d = date.fromisoformat(start)
    end_d = date.fromisoformat(end)
    sources = list(SOURCES) if all_sources else [source]

    for src in sources:
        download_firms(
            start=start_d,
            end=end_d,
            source=src,
            map_key=cfg.firms_map_key,
            bbox=cfg.peten_bbox,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    app()

"""Descarga datos climáticos diarios de NASA POWER para la región del Petén.

Usa el endpoint regional que retorna todos los puntos de la grilla 0.5 x 0.5 grados
dentro del bbox del Petén. Descarga un parámetro por request (límite actual de la API).

Uso:
    uv run python -m src.data.download_power --start 2018-01-01 --end 2024-12-31
    uv run python -m src.data.download_power --year 2023
"""

from __future__ import annotations

import sys
import time
from datetime import date
from io import StringIO
from pathlib import Path
from typing import Annotated

import httpx
import pandas as pd
import typer
from loguru import logger

from src.data.config import settings as default_settings

app = typer.Typer(add_completion=False)

BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/regional"
# La API regional actualmente admite 1 parámetro por request
PARAMETERS = ["T2M", "T2M_MAX", "T2M_MIN", "RH2M", "WS10M", "PRECTOTCORR"]
COMMUNITY = "RE"
# La API exige al menos 2° de rango en latitud — expandimos el bbox ligeramente
LAT_MIN_EXPANDED = 15.82  # bbox original 15.88 → ampliado para cumplir el mínimo
LAT_MAX_EXPANDED = 17.88  # bbox original 17.82 → ampliado


def _parse_power_csv(raw: str) -> pd.DataFrame:
    """Parsea el formato CSV de NASA POWER (con sección de metadatos).

    Args:
        raw: Texto crudo de la respuesta CSV.

    Returns:
        DataFrame con columnas lat, lon, date y el parámetro descargado.
    """
    lines = raw.splitlines()
    data_start = 0
    for i, line in enumerate(lines):
        if line.strip().upper() == "-END HEADER-":
            data_start = i + 1
            break

    data_text = "\n".join(lines[data_start:]).strip()
    if not data_text:
        return pd.DataFrame()

    df = pd.read_csv(StringIO(data_text), skipinitialspace=True)
    df.columns = df.columns.str.strip()

    if {"YEAR", "MO", "DY"}.issubset(df.columns):
        df["date"] = pd.to_datetime(
            df[["YEAR", "MO", "DY"]].rename(columns={"YEAR": "year", "MO": "month", "DY": "day"})
        )
        df = df.drop(columns=["YEAR", "MO", "DY"])

    df = df.rename(columns={"LAT": "lat", "LON": "lon"})

    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].replace(-999.0, float("nan"))

    return df


def _fetch_parameter(
    param: str,
    year: int,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    max_retries: int = 3,
) -> pd.DataFrame | None:
    """Descarga un único parámetro POWER para un año completo.

    Args:
        param: Nombre del parámetro POWER (ej. 'T2M').
        year: Año a descargar.
        lon_min: Longitud mínima del bbox.
        lon_max: Longitud máxima del bbox.
        lat_min: Latitud mínima del bbox (debe dar ≥2° de rango).
        lat_max: Latitud máxima del bbox.
        max_retries: Reintentos ante fallos de red.

    Returns:
        DataFrame con columnas [lat, lon, date, {param}], o None si falló.
    """
    start_str = f"{year}0101"
    end_str = f"{year}1231"

    params = {
        "parameters": param,
        "community": COMMUNITY,
        "longitude-min": lon_min,
        "longitude-max": lon_max,
        "latitude-min": lat_min,
        "latitude-max": lat_max,
        "start": start_str,
        "end": end_str,
        "format": "CSV",
    }

    for attempt in range(1, max_retries + 1):
        try:
            resp = httpx.get(BASE_URL, params=params, timeout=180, follow_redirects=True)
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
            logger.error(f"HTTP {resp.status_code} para {param} {year}: {resp.text[:300]}")
            return None

        try:
            df = _parse_power_csv(resp.text)
        except Exception as exc:
            logger.error(f"Error parseando CSV {param} {year}: {exc}")
            return None

        if df.empty:
            logger.warning(f"POWER {param} {year}: respuesta vacía")
            return None

        logger.debug(f"  {param} {year}: {len(df):,} filas descargadas")
        return df

    return None


def download_power_year(
    year: int,
    bbox: tuple[float, float, float, float],
    output_dir: Path,
    max_retries: int = 3,
) -> Path | None:
    """Descarga todos los parámetros POWER para un año y los guarda en parquet.

    Descarga cada parámetro por separado (límite actual de la API regional) y
    los une en un único DataFrame por (lat, lon, date).

    Args:
        year: Año a descargar.
        bbox: Bounding box (lon_min, lat_min, lon_max, lat_max) en WGS84.
        output_dir: Directorio de salida.
        max_retries: Reintentos por parámetro.

    Returns:
        Ruta al parquet generado, o None si algún parámetro crítico falló.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"power_{year}.parquet"

    if output_path.exists():
        logger.info(f"Ya existe: {output_path} — saltando")
        return output_path

    lon_min, lat_min, lon_max, lat_max = bbox
    # Expandir lat para cumplir el mínimo de 2° que exige la API
    lat_min_exp = min(lat_min, LAT_MIN_EXPANDED)
    lat_max_exp = max(lat_max, LAT_MAX_EXPANDED)

    if lat_max_exp - lat_min_exp < 2.0:
        lat_min_exp -= 0.1
        lat_max_exp += 0.1

    logger.info(f"Descargando POWER {year} ({len(PARAMETERS)} parámetros)...")

    dfs: list[pd.DataFrame] = []
    for param in PARAMETERS:
        df = _fetch_parameter(
            param, year, lon_min, lon_max, lat_min_exp, lat_max_exp, max_retries
        )
        if df is None:
            logger.error(f"No se pudo descargar {param} para {year}")
            return None
        # Mantener solo columnas clave + parámetro
        keep = ["lat", "lon", "date", param]
        dfs.append(df[[c for c in keep if c in df.columns]])
        time.sleep(1.5)  # respetar rate limits entre parámetros

    # Unir todos los parámetros en un solo DataFrame
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on=["lat", "lon", "date"], how="outer")

    merged.to_parquet(output_path, index=False, compression="snappy")
    logger.success(
        f"POWER {year}: {len(merged):,} filas, {merged['lat'].nunique()} puntos → {output_path}"
    )
    return output_path


def download_power(
    start: date,
    end: date,
    bbox: tuple[float, float, float, float],
    output_dir: Path,
) -> list[Path]:
    """Descarga datos POWER para un rango de fechas, año por año.

    Args:
        start: Fecha de inicio.
        end: Fecha de fin.
        bbox: Bounding box en WGS84.
        output_dir: Directorio de salida.

    Returns:
        Lista de rutas a los archivos parquet generados.
    """
    years = range(start.year, end.year + 1)
    paths = []
    for year in years:
        path = download_power_year(year, bbox, output_dir)
        if path:
            paths.append(path)
        time.sleep(3)  # pausa entre años
    return paths


@app.command()
def main(
    start: Annotated[str, typer.Option("--start", help="Fecha inicio YYYY-MM-DD")] = "2018-01-01",
    end: Annotated[str, typer.Option("--end", help="Fecha fin YYYY-MM-DD")] = "2024-12-31",
    year: Annotated[int | None, typer.Option("--year", help="Descargar un año específico")] = None,
    output_dir: Annotated[Path, typer.Option("--output-dir")] = Path("data/raw/power"),
    log_level: Annotated[str, typer.Option("--log-level")] = "INFO",
) -> None:
    """Descarga datos climáticos diarios NASA POWER para el Petén."""
    logger.remove()
    logger.add(sys.stderr, level=log_level.upper())

    cfg = default_settings
    bbox = cfg.peten_bbox

    if year is not None:
        download_power_year(year, bbox, output_dir)
    else:
        start_d = date.fromisoformat(start)
        end_d = date.fromisoformat(end)
        paths = download_power(start_d, end_d, bbox, output_dir)
        logger.info(f"Total: {len(paths)} archivos descargados")


if __name__ == "__main__":
    app()

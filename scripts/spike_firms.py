"""TASK-03 [GATE] — Spike NASA FIRMS API.

Valida que el endpoint responde, que la MAP key es válida y que el CSV
tiene las columnas esperadas para el bbox del Petén.

Uso:
    uv run python scripts/spike_firms.py
"""

from __future__ import annotations

import sys

import httpx

sys.path.insert(0, ".")
from src.data.config import settings  # noqa: E402

# VIIRS y MODIS tienen schemas distintos — columna de brillo difiere
REQUIRED_COLUMNS_BY_SOURCE: dict[str, set[str]] = {
    "VIIRS_SNPP_NRT":    {"latitude", "longitude", "acq_date", "acq_time", "confidence", "bright_ti4"},
    "VIIRS_NOAA20_NRT":  {"latitude", "longitude", "acq_date", "acq_time", "confidence", "bright_ti4"},
    "MODIS_NRT":         {"latitude", "longitude", "acq_date", "acq_time", "confidence", "brightness"},
}
SOURCES = list(REQUIRED_COLUMNS_BY_SOURCE.keys())


def spike_source(source: str, bbox: tuple[float, float, float, float], map_key: str) -> bool:
    """Consulta 1 día de datos para una fuente y verifica la respuesta."""
    lon_min, lat_min, lon_max, lat_max = bbox
    bbox_str = f"{lon_min},{lat_min},{lon_max},{lat_max}"
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{map_key}/{source}/{bbox_str}/1"
    expected = REQUIRED_COLUMNS_BY_SOURCE[source]

    print(f"\n  Consultando {source}...")
    print(f"  URL: {url[:80]}...")

    try:
        resp = httpx.get(url, timeout=30, follow_redirects=True)
    except httpx.RequestError as exc:
        print(f"  ✗ Error de conexión: {exc}")
        return False

    if resp.status_code != 200:
        print(f"  ✗ HTTP {resp.status_code}: {resp.text[:200]}")
        return False

    lines = resp.text.strip().splitlines()
    if not lines:
        print("  ✗ Respuesta vacía")
        return False

    header = set(lines[0].lower().split(","))
    missing = expected - header
    if missing:
        print(f"  ✗ Columnas faltantes: {missing}")
        print(f"  Header recibido: {lines[0][:120]}")
        return False

    n_rows = len(lines) - 1
    print(f"  ✓ HTTP 200 — {n_rows} focos en las últimas 24h  |  cols={len(header)}")
    if n_rows > 0:
        print(f"  Primera fila: {lines[1][:100]}")
    else:
        print("  (sin focos activos en el período — normal fuera de temporada seca)")

    return True


def main() -> None:
    print("=" * 60)
    print("Spike: NASA FIRMS API")
    print("=" * 60)

    if not settings.firms_map_key:
        print("✗ FIRMS_MAP_KEY no está configurada en .env.local")
        sys.exit(1)

    print(f"MAP key: {settings.firms_map_key[:6]}***")
    print(f"Bbox Petén: {settings.peten_bbox}")

    results = [spike_source(src, settings.peten_bbox, settings.firms_map_key) for src in SOURCES]

    print("\n" + "=" * 60)
    if all(results):
        print("✓ FIRMS OK — spike pasado. TASK-03 desbloqueado.")
        sys.exit(0)
    else:
        print("✗ FIRMS FAIL — revisar MAP key o conectividad.")
        sys.exit(1)


if __name__ == "__main__":
    main()

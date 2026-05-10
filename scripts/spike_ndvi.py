"""TASK-05 [GATE] — Spike MODIS NDVI vía NASA Earthdata / AppEEARS.

Valida autenticación con Earthdata y acceso a MOD13Q1 para el tile
h09v07 (cubre el departamento del Petén).

Estrategia en dos pasos:
  1. Verificar credenciales Earthdata con el endpoint de autenticación.
  2. Intentar listar granulas de MOD13Q1 disponibles para el bbox del Petén
     vía CMR (Common Metadata Repository) — sin descarga, solo metadatos.

Uso:
    uv run python scripts/spike_ndvi.py
"""

from __future__ import annotations

import sys

import httpx

sys.path.insert(0, ".")
from src.data.config import settings  # noqa: E402

CMR_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
# MOD13Q1.061 — MODIS/Terra Vegetation Indices 16-Day L3 Global 250m
COLLECTION_SHORT_NAME = "MOD13Q1"
COLLECTION_VERSION = "061"

# Tile h09v07 cubre Guatemala/Petén en la proyección sinusoidal MODIS
TILE = "h09v07"


def check_earthdata_auth(user: str, password: str) -> bool:
    """Verifica que las credenciales Earthdata son válidas."""
    print("  Verificando credenciales Earthdata...")
    # URS (Earthdata Login) endpoint de perfil — requiere auth básica
    resp = httpx.get(
        "https://urs.earthdata.nasa.gov/api/users/tokens",
        auth=(user, password),
        timeout=20,
        follow_redirects=True,
    )
    if resp.status_code in (200, 201):
        print(f"  ✓ Autenticación OK (HTTP {resp.status_code})")
        return True
    print(f"  ✗ Auth fallida: HTTP {resp.status_code} — {resp.text[:200]}")
    return False


def check_mod13q1_granules() -> bool:
    """Lista granulas MOD13Q1 disponibles para el tile h09v07 (sin descarga)."""
    print(f"\n  Buscando granulas {COLLECTION_SHORT_NAME} v{COLLECTION_VERSION} para tile {TILE}...")

    # CMR acepta readable_granule_name con wildcard para filtrar por tile
    params = {
        "short_name": COLLECTION_SHORT_NAME,
        "version": COLLECTION_VERSION,
        "temporal[]": "2024-01-01T00:00:00Z,2024-03-31T23:59:59Z",
        "readable_granule_name[]": f"*{TILE}*",
        "page_size": 5,
    }
    try:
        resp = httpx.get(CMR_URL, params=params, timeout=30)
    except httpx.RequestError as exc:
        print(f"  ✗ Error de conexión a CMR: {exc}")
        return False

    if resp.status_code != 200:
        print(f"  ✗ CMR HTTP {resp.status_code}: {resp.text[:200]}")
        return False

    data = resp.json()
    entries = data.get("feed", {}).get("entry", [])

    if not entries:
        # Fallback: buscar sin filtro de tile para verificar que la colección existe
        print("  ! Sin resultados para el tile — verificando que la colección existe...")
        params_fallback = {
            "short_name": COLLECTION_SHORT_NAME,
            "version": COLLECTION_VERSION,
            "temporal[]": "2024-01-01T00:00:00Z,2024-01-31T23:59:59Z",
            "page_size": 3,
        }
        resp2 = httpx.get(CMR_URL, params=params_fallback, timeout=30)
        entries2 = resp2.json().get("feed", {}).get("entry", []) if resp2.status_code == 200 else []
        if entries2:
            print(f"  ✓ Colección existe ({len(entries2)} granulas en enero 2024)")
            print(f"    Ejemplo: {entries2[0].get('title', 'N/A')}")
            print(f"  ! Tile {TILE} no retornó resultados con readable_granule_name — usar bbox en descarga")
            return True
        print("  ✗ Colección no encontrada en CMR. Verificar short_name/version.")
        return False

    print(f"  ✓ {len(entries)} granulas para tile {TILE}:")
    for g in entries[:3]:
        title = g.get("title", g.get("producer_granule_id", "N/A"))
        time_start = g.get("time_start", "N/A")[:10]
        print(f"    - {title}  ({time_start})")

    # Verificar que hay URL de descarga disponible
    links = entries[0].get("links", [])
    dl_rel = "http://esipfed.org/ns/fedsearch/1.1/data#"
    download_links = [lk["href"] for lk in links if lk.get("rel") == dl_rel]
    if download_links:
        print(f"  ✓ URL de descarga: {download_links[0][:80]}...")
    else:
        print("  ! Sin link directo — se usará AppEEARS o GEE (ver ARCHITECTURE.md §2.2)")

    return True


def main() -> None:
    print("=" * 60)
    print("Spike: MODIS MOD13Q1 — NASA Earthdata")
    print("=" * 60)

    if not settings.earthdata_user or not settings.earthdata_pass:
        print("✗ EARTHDATA_USER / EARTHDATA_PASS no configuradas en .env.local")
        sys.exit(1)

    print(f"Usuario: {settings.earthdata_user}")

    auth_ok = check_earthdata_auth(settings.earthdata_user, settings.earthdata_pass)
    granules_ok = check_mod13q1_granules()

    print("\n" + "=" * 60)
    if auth_ok and granules_ok:
        print("✓ MODIS NDVI OK — spike pasado. TASK-05 desbloqueado.")
        print("  Estrategia de descarga recomendada: CMR + httpx con auth Earthdata")
        sys.exit(0)
    elif not auth_ok:
        print("✗ Credenciales Earthdata inválidas. Verificar user/pass en .env.local")
        sys.exit(1)
    else:
        print("⚠ Auth OK pero granulas no encontradas. Revisar versión de colección.")
        print("  Alternativa: Google Earth Engine (ver ARCHITECTURE.md §2.2)")
        sys.exit(1)


if __name__ == "__main__":
    main()

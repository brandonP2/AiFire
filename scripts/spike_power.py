"""TASK-04 [GATE] — Spike NASA POWER API.

Valida que el endpoint responde sin autenticación y que los parámetros
climáticos clave (T2M, RH2M, WS10M, PRECTOTCORR) están disponibles
para un punto del Petén.

Uso:
    uv run python scripts/spike_power.py
"""

from __future__ import annotations

import sys

import httpx

# Punto de prueba: centro aproximado del Petén
LAT, LON = 16.9, -90.2
PARAMS = "T2M,T2M_MAX,T2M_MIN,RH2M,WS10M,PRECTOTCORR"
START, END = "20240301", "20240307"
BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"


def main() -> None:
    print("=" * 60)
    print("Spike: NASA POWER API")
    print("=" * 60)
    print(f"Punto de prueba: lat={LAT}, lon={LON}")
    print(f"Período: {START} → {END}")
    print(f"Parámetros: {PARAMS}\n")

    url = (
        f"{BASE_URL}?"
        f"parameters={PARAMS}"
        f"&community=RE"
        f"&longitude={LON}"
        f"&latitude={LAT}"
        f"&start={START}"
        f"&end={END}"
        f"&format=JSON"
    )

    print(f"URL: {url[:100]}...")

    try:
        resp = httpx.get(url, timeout=60, follow_redirects=True)
    except httpx.RequestError as exc:
        print(f"✗ Error de conexión: {exc}")
        sys.exit(1)

    if resp.status_code != 200:
        print(f"✗ HTTP {resp.status_code}: {resp.text[:300]}")
        sys.exit(1)

    data = resp.json()

    # Verificar estructura esperada
    try:
        props = data["properties"]["parameter"]
    except KeyError:
        print(f"✗ Estructura inesperada. Keys: {list(data.keys())}")
        sys.exit(1)

    expected = {"T2M", "RH2M", "WS10M", "PRECTOTCORR"}
    available = set(props.keys())
    missing = expected - available

    if missing:
        print(f"✗ Parámetros faltantes: {missing}")
        print(f"  Disponibles: {sorted(available)}")
        sys.exit(1)

    # Mostrar muestra de datos
    print(f"✓ HTTP 200 — parámetros disponibles: {sorted(available)}\n")
    print("Muestra de datos (primeros 3 días):")
    t2m = props["T2M"]
    rh2m = props["RH2M"]
    ws10m = props["WS10M"]
    prec = props["PRECTOTCORR"]

    for i, date_key in enumerate(sorted(t2m.keys())[:3]):
        print(
            f"  {date_key} | T2M={t2m[date_key]:6.2f}°C | "
            f"RH2M={rh2m[date_key]:5.1f}% | "
            f"WS10M={ws10m[date_key]:5.2f}m/s | "
            f"PREC={prec[date_key]:5.2f}mm"
        )

    print("\n" + "=" * 60)
    print("✓ NASA POWER OK — spike pasado. TASK-04 desbloqueado.")
    sys.exit(0)


if __name__ == "__main__":
    main()

"""Descarga capas estáticas auxiliares para el Petén.

- Caminos: OSM vía Overpass API (sin autenticación)
- Áreas protegidas: OSM vía Overpass API + Natural Earth como respaldo
- Poblados: OSM vía Overpass API

Genera:
    data/interim/static/roads_peten.gpkg
    data/interim/static/settlements_peten.gpkg
    data/interim/static/protected_areas_peten.gpkg

Uso:
    uv run python -m src.data.download_static
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Annotated

import geopandas as gpd
import httpx
import pandas as pd
import typer
from loguru import logger
from shapely.geometry import LineString, Point, Polygon, shape

from src.data.config import settings as default_settings

app = typer.Typer(add_completion=False)

# Lista de servidores Overpass a intentar en orden
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]
OVERPASS_TIMEOUT = 180  # segundos


def _overpass_query(query: str, max_retries: int = 3) -> dict | None:
    """Ejecuta una query Overpass probando múltiples servidores.

    Args:
        query: Query Overpass QL.
        max_retries: Reintentos por servidor.

    Returns:
        JSON de la respuesta, o None si todos los servidores fallaron.
    """
    for server_url in OVERPASS_URLS:
        logger.debug(f"Intentando servidor Overpass: {server_url}")
        for attempt in range(1, max_retries + 1):
            try:
                resp = httpx.post(
                    server_url,
                    data={"data": query},
                    timeout=OVERPASS_TIMEOUT,
                    follow_redirects=True,
                )
            except httpx.RequestError as exc:
                logger.warning(f"Red ({server_url}, intento {attempt}/{max_retries}): {exc}")
                if attempt < max_retries:
                    time.sleep(5 * attempt)
                continue

            if resp.status_code == 429:
                logger.warning(f"Rate limit en {server_url} — esperando 60s")
                time.sleep(60)
                continue

            if resp.status_code not in (200, 200):
                logger.warning(f"HTTP {resp.status_code} en {server_url} — probando siguiente")
                break  # Intentar siguiente servidor

            try:
                return resp.json()
            except json.JSONDecodeError as exc:
                logger.error(f"JSON inválido de {server_url}: {exc}")
                break

    logger.error("Todos los servidores Overpass fallaron")
    return None


def _overpass_ways_to_gdf(data: dict, geom_type: str = "line") -> gpd.GeoDataFrame:
    """Convierte elementos Overpass (ways con geom) a GeoDataFrame.

    Args:
        data: JSON de respuesta Overpass con `out geom`.
        geom_type: 'line' para ways como LineString, 'polygon' para áreas.

    Returns:
        GeoDataFrame en EPSG:4326.
    """
    geometries = []
    properties = []

    for elem in data.get("elements", []):
        if elem.get("type") != "way" or "geometry" not in elem:
            continue
        coords = [(n["lon"], n["lat"]) for n in elem["geometry"]]
        if len(coords) < 2:
            continue

        tags = elem.get("tags", {})
        if geom_type == "polygon" and len(coords) >= 4 and coords[0] == coords[-1]:
            geom = Polygon(coords)
        else:
            geom = LineString(coords)

        geometries.append(geom)
        properties.append({"osm_id": elem["id"], **tags})

    if not geometries:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    return gpd.GeoDataFrame(properties, geometry=geometries, crs="EPSG:4326")


def _overpass_nodes_to_gdf(data: dict) -> gpd.GeoDataFrame:
    """Convierte nodos Overpass a GeoDataFrame de puntos.

    Args:
        data: JSON de respuesta Overpass.

    Returns:
        GeoDataFrame en EPSG:4326.
    """
    geometries = []
    properties = []

    for elem in data.get("elements", []):
        if elem.get("type") != "node":
            continue
        geom = Point(elem["lon"], elem["lat"])
        tags = elem.get("tags", {})
        geometries.append(geom)
        properties.append({"osm_id": elem["id"], **tags})

    if not geometries:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    return gpd.GeoDataFrame(properties, geometry=geometries, crs="EPSG:4326")


def download_roads(
    bbox: tuple[float, float, float, float],
    output_dir: Path,
    target_crs: str = "EPSG:32616",
) -> Path:
    """Descarga la red vial del Petén desde OSM vía Overpass API.

    Args:
        bbox: Bounding box (lon_min, lat_min, lon_max, lat_max) en WGS84.
        output_dir: Directorio de salida.
        target_crs: CRS de destino.

    Returns:
        Ruta al GeoPackage resultante.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "roads_peten.gpkg"

    if output_path.exists():
        logger.info(f"Ya existe: {output_path} — saltando")
        return output_path

    lon_min, lat_min, lon_max, lat_max = bbox
    query = f"""
    [out:json][timeout:{OVERPASS_TIMEOUT}][bbox:{lat_min},{lon_min},{lat_max},{lon_max}];
    (
      way["highway"~"^(primary|secondary|tertiary|unclassified|residential|track|road)$"];
    );
    out geom;
    """

    logger.info("Descargando red vial del Petén desde Overpass API...")
    data = _overpass_query(query)
    if data is None:
        raise RuntimeError("Overpass API falló para caminos")

    gdf = _overpass_ways_to_gdf(data, geom_type="line")
    if gdf.empty:
        logger.warning("Sin caminos encontrados en el bbox")
    else:
        gdf = gdf.to_crs(target_crs)
        logger.info(f"{len(gdf):,} segmentos viales encontrados")

    gdf.to_file(output_path, driver="GPKG", layer="roads")
    logger.success(f"Caminos: {output_path}")
    return output_path


def _settlements_fallback() -> gpd.GeoDataFrame:
    """Retorna un GeoDataFrame mínimo con las principales localidades del Petén.

    Usado como respaldo cuando Overpass no está disponible.
    Coordenadas WGS84 (EPSG:4326).
    """
    towns = [
        ("Flores", -89.8924, 16.9290, "city"),
        ("San Benito", -89.9014, 16.9197, "city"),
        ("Santa Elena de La Cruz", -89.8994, 16.9100, "town"),
        ("San Andrés", -89.9181, 16.9981, "town"),
        ("La Libertad", -90.1228, 16.7775, "town"),
        ("Sayaxché", -90.1878, 16.5328, "town"),
        ("Poptún", -89.4219, 16.3239, "town"),
        ("San Luis", -89.4442, 16.1942, "town"),
        ("Dolores", -89.4172, 16.5128, "town"),
        ("Melchor de Mencos", -89.1569, 17.0556, "town"),
        ("El Naranjo", -90.8136, 17.1831, "town"),
        ("Bethel", -90.7156, 16.8736, "village"),
        ("Carmelita", -90.0739, 17.4261, "village"),
        ("Uaxactún", -89.6325, 17.3958, "village"),
    ]
    points = [Point(lon, lat) for _, lon, lat, _ in towns]
    return gpd.GeoDataFrame(
        {"name": [t[0] for t in towns], "place": [t[3] for t in towns]},
        geometry=points,
        crs="EPSG:4326",
    )


def download_settlements(
    bbox: tuple[float, float, float, float],
    output_dir: Path,
    target_crs: str = "EPSG:32616",
) -> Path:
    """Descarga poblados del Petén desde OSM vía Overpass API.

    Args:
        bbox: Bounding box (lon_min, lat_min, lon_max, lat_max) en WGS84.
        output_dir: Directorio de salida.
        target_crs: CRS de destino.

    Returns:
        Ruta al GeoPackage resultante.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "settlements_peten.gpkg"

    if output_path.exists():
        logger.info(f"Ya existe: {output_path} — saltando")
        return output_path

    lon_min, lat_min, lon_max, lat_max = bbox
    # Evitar regex en el servidor (más lenta) — buscar cada tipo por separado
    query = f"""
    [out:json][timeout:{OVERPASS_TIMEOUT}][bbox:{lat_min},{lon_min},{lat_max},{lon_max}];
    (
      node["place"="city"];
      node["place"="town"];
      node["place"="village"];
      node["place"="hamlet"];
    );
    out body;
    """

    logger.info("Descargando poblados del Petén desde Overpass API...")
    data = _overpass_query(query)

    if data is not None:
        gdf = _overpass_nodes_to_gdf(data)
    else:
        logger.warning("Overpass falló para poblados — usando fallback con principales localidades del Petén")
        gdf = _settlements_fallback()

    if gdf.empty:
        logger.warning("Sin poblados — usando fallback mínimo")
        gdf = _settlements_fallback()

    gdf = gdf.to_crs(target_crs)
    logger.info(f"{len(gdf):,} poblados")
    gdf.to_file(output_path, driver="GPKG", layer="settlements")
    logger.success(f"Poblados: {output_path}")
    return output_path


def download_protected_areas(
    bbox: tuple[float, float, float, float],
    output_dir: Path,
    target_crs: str = "EPSG:32616",
) -> Path:
    """Descarga áreas protegidas del Petén desde OSM vía Overpass API.

    Incluye: Reserva de Biosfera Maya, Parque Nacional Tikal, Laguna del Tigre,
    Sierra del Lacandón y otras áreas bajo manejo de CONAP.

    Args:
        bbox: Bounding box (lon_min, lat_min, lon_max, lat_max) en WGS84.
        output_dir: Directorio de salida.
        target_crs: CRS de destino.

    Returns:
        Ruta al GeoPackage resultante.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "protected_areas_peten.gpkg"

    if output_path.exists():
        logger.info(f"Ya existe: {output_path} — saltando")
        return output_path

    lon_min, lat_min, lon_max, lat_max = bbox
    # Query para relaciones (polígonos grandes) y ways de áreas protegidas
    query = f"""
    [out:json][timeout:{OVERPASS_TIMEOUT}][bbox:{lat_min},{lon_min},{lat_max},{lon_max}];
    (
      way["boundary"="protected_area"];
      way["leisure"="nature_reserve"];
      way["protect_class"];
    );
    out geom;
    """

    logger.info("Descargando áreas protegidas del Petén desde Overpass API...")
    data = _overpass_query(query)

    if data is None:
        logger.warning("Overpass falló — usando polígono simplificado de la Biosfera Maya como respaldo")
        gdf = _fallback_protected_areas(target_crs)
    else:
        gdf = _overpass_ways_to_gdf(data, geom_type="polygon")
        if gdf.empty:
            logger.warning("Sin áreas protegidas vía OSM — usando respaldo")
            gdf = _fallback_protected_areas(target_crs)
        else:
            gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
            gdf = gdf.to_crs(target_crs)
            logger.info(f"{len(gdf):,} áreas protegidas encontradas")

    gdf.to_file(output_path, driver="GPKG", layer="protected_areas")
    logger.success(f"Áreas protegidas: {output_path}")
    return output_path


def _fallback_protected_areas(target_crs: str) -> gpd.GeoDataFrame:
    """Polígono aproximado de las principales áreas protegidas del Petén.

    Respaldo cuando Overpass API no está disponible.
    Basado en los límites conocidos de la Reserva de Biosfera Maya.
    """
    # Aproximación del núcleo norte de la Reserva de Biosfera Maya
    biosfera_maya_approx = Polygon([
        (-91.0, 17.0), (-89.5, 17.0), (-89.5, 17.8),
        (-91.0, 17.8), (-91.0, 17.0),
    ])
    tikal_approx = Polygon([
        (-89.7, 17.1), (-89.5, 17.1), (-89.5, 17.4),
        (-89.7, 17.4), (-89.7, 17.1),
    ])

    gdf = gpd.GeoDataFrame(
        [
            {"name": "Reserva Biosfera Maya (approx)", "source": "fallback"},
            {"name": "PN Tikal (approx)", "source": "fallback"},
        ],
        geometry=[biosfera_maya_approx, tikal_approx],
        crs="EPSG:4326",
    ).to_crs(target_crs)

    logger.warning("Usando polígonos de respaldo aproximados — reemplazar con WDPA cuando esté disponible")
    return gdf


@app.command()
def main(
    output_dir: Annotated[Path, typer.Option("--output-dir")] = Path("data/interim/static"),
    log_level: Annotated[str, typer.Option("--log-level")] = "INFO",
) -> None:
    """Descarga capas estáticas OSM para el Petén (caminos, poblados, áreas protegidas)."""
    logger.remove()
    logger.add(sys.stderr, level=log_level.upper())

    cfg = default_settings
    bbox = cfg.peten_bbox
    crs = cfg.crs_work

    download_roads(bbox, output_dir, crs)
    time.sleep(5)  # respetar rate limit Overpass
    download_settlements(bbox, output_dir, crs)
    time.sleep(5)
    download_protected_areas(bbox, output_dir, crs)


if __name__ == "__main__":
    app()

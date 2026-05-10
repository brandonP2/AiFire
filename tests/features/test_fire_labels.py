"""Tests de src.features.fire_labels."""

from __future__ import annotations

from datetime import date

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from src.features.fire_labels import assign_fires_to_grid


@pytest.fixture()
def tiny_grid() -> gpd.GeoDataFrame:
    """Grilla de 2 celdas en EPSG:32616."""
    cells = [
        box(500_000, 1_800_000, 501_000, 1_801_000),
        box(501_000, 1_800_000, 502_000, 1_801_000),
    ]
    return gpd.GeoDataFrame(
        {"cell_id": ["r0000_c0000", "r0000_c0001"]},
        geometry=cells,
        crs="EPSG:32616",
    )


def test_assign_fires_single_fire(tiny_grid: gpd.GeoDataFrame) -> None:
    """Un foco dentro de la primera celda debe asignarse a esa celda."""
    from pyproj import Transformer

    # Convertir el centroide de la primera celda a WGS84
    t = Transformer.from_crs("EPSG:32616", "EPSG:4326", always_xy=True)
    lon, lat = t.transform(500_500.0, 1_800_500.0)

    firms_df = pd.DataFrame(
        {
            "latitude": [lat],
            "longitude": [lon],
            "acq_date": [date(2023, 3, 15)],
            "confidence": ["nominal"],
        }
    )

    result = assign_fires_to_grid(firms_df, tiny_grid)

    assert len(result) == 1
    assert result.iloc[0]["cell_id"] == "r0000_c0000"
    assert result.iloc[0]["fire_occurred"]


def test_assign_fires_no_fires(tiny_grid: gpd.GeoDataFrame) -> None:
    """Sin focos, el resultado debe ser un DataFrame vacío."""
    firms_df = pd.DataFrame(
        columns=["latitude", "longitude", "acq_date", "confidence"]
    )
    result = assign_fires_to_grid(firms_df, tiny_grid)
    assert len(result) == 0


def test_assign_fires_outside_grid(tiny_grid: gpd.GeoDataFrame) -> None:
    """Focos fuera del bbox de la grilla no deben aparecer en el resultado."""
    firms_df = pd.DataFrame(
        {
            "latitude": [20.0],     # fuera del Petén
            "longitude": [-85.0],
            "acq_date": [date(2023, 4, 1)],
            "confidence": ["nominal"],
        }
    )
    result = assign_fires_to_grid(firms_df, tiny_grid)
    assert len(result) == 0


def test_assign_fires_multiple_fires_same_cell(tiny_grid: gpd.GeoDataFrame) -> None:
    """Múltiples focos en la misma celda/día deben colapsar en una sola fila fire_occurred=True."""
    from pyproj import Transformer

    t = Transformer.from_crs("EPSG:32616", "EPSG:4326", always_xy=True)
    lon1, lat1 = t.transform(500_300.0, 1_800_300.0)
    lon2, lat2 = t.transform(500_700.0, 1_800_700.0)

    firms_df = pd.DataFrame(
        {
            "latitude": [lat1, lat2],
            "longitude": [lon1, lon2],
            "acq_date": [date(2023, 3, 20), date(2023, 3, 20)],
            "confidence": ["nominal", "high"],
        }
    )

    result = assign_fires_to_grid(firms_df, tiny_grid)

    # Solo debe haber 1 fila para (r0000_c0000, 2023-03-20)
    assert len(result) == 1
    assert result.iloc[0]["cell_id"] == "r0000_c0000"


def test_assign_fires_different_cells(tiny_grid: gpd.GeoDataFrame) -> None:
    """Focos en diferentes celdas deben generar filas separadas."""
    from pyproj import Transformer

    t = Transformer.from_crs("EPSG:32616", "EPSG:4326", always_xy=True)
    lon0, lat0 = t.transform(500_500.0, 1_800_500.0)
    lon1, lat1 = t.transform(501_500.0, 1_800_500.0)

    firms_df = pd.DataFrame(
        {
            "latitude": [lat0, lat1],
            "longitude": [lon0, lon1],
            "acq_date": [date(2023, 4, 5), date(2023, 4, 5)],
            "confidence": ["nominal", "nominal"],
        }
    )

    result = assign_fires_to_grid(firms_df, tiny_grid)
    assert len(result) == 2
    assert set(result["cell_id"]) == {"r0000_c0000", "r0000_c0001"}

"""Tests de src.features.static_features."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import rasterio.transform
from shapely.geometry import box

from src.features.static_features import (
    _sample_raster_at_centroids,
    compute_protected_areas,
    compute_topo_features,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tiny_grid() -> gpd.GeoDataFrame:
    """Grilla mínima de 3 celdas en EPSG:32616."""
    cells = [
        box(500_000, 1_800_000, 501_000, 1_801_000),
        box(501_000, 1_800_000, 502_000, 1_801_000),
        box(502_000, 1_800_000, 503_000, 1_801_000),
    ]
    return gpd.GeoDataFrame(
        {
            "cell_id": ["r0000_c0000", "r0000_c0001", "r0000_c0002"],
            "row": [0, 0, 0],
            "col": [0, 1, 2],
            "lon_center": [-90.5, -90.49, -90.48],
            "lat_center": [16.3, 16.3, 16.3],
            "easting_m": [500_500.0, 501_500.0, 502_500.0],
            "northing_m": [1_800_500.0, 1_800_500.0, 1_800_500.0],
        },
        geometry=cells,
        crs="EPSG:32616",
    )


@pytest.fixture()
def fake_raster(tmp_path: Path) -> Path:
    """GeoTIFF de un solo canal con valores conocidos (EPSG:32616)."""
    raster_path = tmp_path / "fake_dem.tif"
    data = np.array([[100.0, 200.0, 300.0]], dtype=np.float32)

    transform = rasterio.transform.from_bounds(
        west=500_000, south=1_800_000,
        east=503_000, north=1_801_000,
        width=3, height=1,
    )
    with rasterio.open(
        raster_path, "w",
        driver="GTiff",
        height=1, width=3,
        count=1,
        dtype="float32",
        crs="EPSG:32616",
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)
    return raster_path


@pytest.fixture()
def fake_protected_areas(tmp_path: Path, tiny_grid: gpd.GeoDataFrame) -> Path:
    """GeoPackage con un área protegida que cubre solo la primera celda."""
    protected_path = tmp_path / "protected.gpkg"
    protected_geom = box(499_000, 1_799_000, 501_500, 1_801_500)
    gdf = gpd.GeoDataFrame(
        {"name": ["Reserva Test"]},
        geometry=[protected_geom],
        crs="EPSG:32616",
    )
    gdf.to_file(protected_path, driver="GPKG")
    return protected_path


# ---------------------------------------------------------------------------
# Tests de _sample_raster_at_centroids
# ---------------------------------------------------------------------------


def test_sample_raster_known_values(fake_raster: Path, tiny_grid: gpd.GeoDataFrame) -> None:
    centroids = tiny_grid.geometry.centroid
    xy = np.column_stack([centroids.x.values, centroids.y.values])
    values = _sample_raster_at_centroids(fake_raster, xy)

    assert len(values) == 3
    assert not np.isnan(values).any(), "No deberían haber NaN para centroides dentro del raster"
    # Los valores deben ser 100, 200, 300 (muestreo por columna)
    np.testing.assert_allclose(values, [100.0, 200.0, 300.0], atol=1.0)


def test_sample_raster_out_of_bounds(fake_raster: Path) -> None:
    """Puntos fuera del raster deben retornar NaN."""
    xy = np.array([[999_999.0, 999_999.0]])
    values = _sample_raster_at_centroids(fake_raster, xy)
    assert np.isnan(values[0])


# ---------------------------------------------------------------------------
# Tests de compute_topo_features
# ---------------------------------------------------------------------------


def test_compute_topo_features(
    tmp_path: Path,
    tiny_grid: gpd.GeoDataFrame,
    fake_raster: Path,
) -> None:
    # Crear los tres rasters esperados con el mismo contenido
    topo_dir = tmp_path / "topography"
    topo_dir.mkdir()
    for name in ("srtm_peten.tif", "slope_peten.tif", "aspect_peten.tif"):
        import shutil
        shutil.copy(fake_raster, topo_dir / name)

    df = compute_topo_features(tiny_grid, topo_dir)

    assert list(df.columns) == ["cell_id", "elevation_m", "slope_deg", "aspect_deg"]
    assert len(df) == 3
    assert (df["cell_id"] == tiny_grid["cell_id"].values).all()
    assert not df["elevation_m"].isna().any()


def test_compute_topo_features_missing_raster(
    tmp_path: Path,
    tiny_grid: gpd.GeoDataFrame,
) -> None:
    topo_dir = tmp_path / "empty_topo"
    topo_dir.mkdir()
    with pytest.raises(FileNotFoundError, match=r"srtm_peten\.tif"):
        compute_topo_features(tiny_grid, topo_dir)


# ---------------------------------------------------------------------------
# Tests de compute_protected_areas
# ---------------------------------------------------------------------------


def test_compute_protected_areas(
    tmp_path: Path,
    tiny_grid: gpd.GeoDataFrame,
    fake_protected_areas: Path,
) -> None:
    df = compute_protected_areas(tiny_grid, fake_protected_areas)

    assert list(df.columns) == ["cell_id", "is_protected_area"]
    assert len(df) == 3
    # La primera celda (x: 500k-501k) intersecta el área protegida
    first_row = df[df["cell_id"] == "r0000_c0000"].iloc[0]
    assert first_row["is_protected_area"]
    # La tercera celda (x: 502k-503k) NO intersecta
    third_row = df[df["cell_id"] == "r0000_c0002"].iloc[0]
    assert not third_row["is_protected_area"]


def test_compute_protected_areas_missing_file(
    tmp_path: Path,
    tiny_grid: gpd.GeoDataFrame,
) -> None:
    with pytest.raises(FileNotFoundError):
        compute_protected_areas(tiny_grid, tmp_path / "nonexistent.gpkg")

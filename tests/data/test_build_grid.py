from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pytest

from src.data.build_grid import _validate_grid, build_grid
from src.data.config import Settings


@pytest.fixture()
def small_cfg() -> Settings:
    """Configuración recortada para tests rápidos (~4x3 celdas)."""
    return Settings(
        peten_bbox=(-90.10, 16.90, -90.06, 16.93),
        grid_resolution_km=1.0,
    )


def test_build_grid_returns_geodataframe(small_cfg: Settings) -> None:
    gdf = build_grid(cfg=small_cfg)
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf) > 0


def test_build_grid_crs_is_utm16n(small_cfg: Settings) -> None:
    gdf = build_grid(cfg=small_cfg)
    assert gdf.crs is not None
    assert gdf.crs.to_epsg() == 32616


def test_build_grid_no_null_geometries(small_cfg: Settings) -> None:
    gdf = build_grid(cfg=small_cfg)
    assert gdf.geometry.isna().sum() == 0


def test_build_grid_all_geometries_valid(small_cfg: Settings) -> None:
    gdf = build_grid(cfg=small_cfg)
    assert gdf.geometry.is_valid.all()


def test_build_grid_cell_area_is_1km2(small_cfg: Settings) -> None:
    gdf = build_grid(cfg=small_cfg)
    areas_km2 = gdf.geometry.area / 1e6
    assert areas_km2.between(0.99, 1.01).all(), f"Áreas fuera de rango: {areas_km2.describe()}"


def test_build_grid_cell_id_unique(small_cfg: Settings) -> None:
    gdf = build_grid(cfg=small_cfg)
    assert not gdf["cell_id"].duplicated().any()


def test_build_grid_cell_id_format(small_cfg: Settings) -> None:
    gdf = build_grid(cfg=small_cfg)
    # Formato esperado: r0000_c0001
    assert gdf["cell_id"].str.match(r"^r\d{4}_c\d{4}$").all()


def test_build_grid_required_columns(small_cfg: Settings) -> None:
    gdf = build_grid(cfg=small_cfg)
    expected = {"cell_id", "row", "col", "lon_center", "lat_center", "easting_m", "northing_m"}
    assert expected.issubset(set(gdf.columns))


def test_build_grid_centroids_within_bbox(small_cfg: Settings) -> None:
    gdf = build_grid(cfg=small_cfg)
    lon_min, lat_min, lon_max, lat_max = small_cfg.peten_bbox
    # Los centroides deben estar dentro del bbox con un margen de 1 km (~0.01°)
    assert gdf["lon_center"].between(lon_min - 0.01, lon_max + 0.01).all()
    assert gdf["lat_center"].between(lat_min - 0.01, lat_max + 0.01).all()


def test_build_grid_saves_to_disk(small_cfg: Settings, tmp_path: Path) -> None:
    out = tmp_path / "test_grid.gpkg"
    gdf = build_grid(cfg=small_cfg, output_path=out)
    assert out.exists()
    loaded = gpd.read_file(out, layer="peten_grid")
    assert len(loaded) == len(gdf)
    assert loaded.crs.to_epsg() == 32616


def test_validate_grid_raises_on_wrong_crs(small_cfg: Settings) -> None:
    gdf = build_grid(cfg=small_cfg)
    gdf_bad = gdf.to_crs("EPSG:4326")
    with pytest.raises(ValueError, match="CRS incorrecto"):
        _validate_grid(gdf_bad)

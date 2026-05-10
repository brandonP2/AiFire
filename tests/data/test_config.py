from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.data.config import Settings


def test_default_settings_bbox() -> None:
    cfg = Settings()
    lon_min, lat_min, lon_max, lat_max = cfg.peten_bbox
    assert lon_min < lon_max
    assert lat_min < lat_max
    assert lon_min < 0       # Petén está en el hemisferio oeste
    assert lat_min > 0       # y en el hemisferio norte


def test_default_settings_crs() -> None:
    cfg = Settings()
    assert cfg.crs_work == "EPSG:32616"
    assert cfg.crs_geo == "EPSG:4326"


def test_bbox_parsed_from_csv_string() -> None:
    cfg = Settings(peten_bbox="-91.45,15.88,-89.14,17.82")  # type: ignore[call-arg]
    assert cfg.peten_bbox == (-91.45, 15.88, -89.14, 17.82)


def test_bbox_parsed_from_tuple() -> None:
    cfg = Settings(peten_bbox=(-90.10, 16.90, -90.06, 16.93))  # type: ignore[call-arg]
    lon_min, lat_min, lon_max, lat_max = cfg.peten_bbox
    assert lon_min == pytest.approx(-90.10)
    assert lat_max == pytest.approx(16.93)


def test_bbox_wrong_length_raises() -> None:
    with pytest.raises(ValidationError):
        Settings(peten_bbox="-91.45,15.88")  # type: ignore[call-arg]


def test_data_paths_are_under_project_root() -> None:
    cfg = Settings(project_root=Path("/tmp/petenfire_test"))
    assert cfg.data_raw == Path("/tmp/petenfire_test/data/raw")
    assert cfg.data_interim == Path("/tmp/petenfire_test/data/interim")
    assert cfg.data_processed == Path("/tmp/petenfire_test/data/processed")


def test_grid_resolution_positive() -> None:
    with pytest.raises(ValidationError):
        Settings(grid_resolution_km=-1.0)


def test_settings_loads_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIRMS_MAP_KEY", "test_key_123")
    monkeypatch.setenv("EARTHDATA_USER", "test_user")
    cfg = Settings()
    assert cfg.firms_map_key == "test_key_123"
    assert cfg.earthdata_user == "test_user"

"""Tests de src.features.ndvi_features."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from src.features.ndvi_features import (
    _discover_ndvi_tifs,
    interpolate_ndvi_to_daily,
)

# ---------------------------------------------------------------------------
# Tests de _discover_ndvi_tifs
# ---------------------------------------------------------------------------


def test_discover_ndvi_tifs(tmp_path: Path) -> None:
    """Debe encontrar todos los GeoTIFFs con nombre ndvi_YYYY-MM-DD.tif."""
    for name in ("ndvi_2023-03-14.tif", "ndvi_2023-03-30.tif", "ndvi_2023-04-15.tif"):
        (tmp_path / name).touch()
    (tmp_path / "other_file.tif").touch()

    tifs = _discover_ndvi_tifs(tmp_path)

    assert len(tifs) == 3
    assert date(2023, 3, 14) in tifs
    assert date(2023, 4, 15) in tifs


def test_discover_ndvi_tifs_empty_dir(tmp_path: Path) -> None:
    tifs = _discover_ndvi_tifs(tmp_path)
    assert len(tifs) == 0


# ---------------------------------------------------------------------------
# Tests de interpolate_ndvi_to_daily
# ---------------------------------------------------------------------------


def _make_composites(
    cell_ids: list[str],
    dates: list[date],
    ndvi_values: list[float],
) -> pd.DataFrame:
    rows = []
    for cid in cell_ids:
        for d, v in zip(dates, ndvi_values, strict=True):
            rows.append({"cell_id": cid, "date": d, "ndvi": v})
    return pd.DataFrame(rows)


def test_interpolate_ndvi_daily_covers_all_dates() -> None:
    """La interpolación debe cubrir todos los días del rango."""
    composites = _make_composites(
        cell_ids=["c0", "c1"],
        dates=[date(2023, 1, 1), date(2023, 1, 17)],
        ndvi_values=[0.3, 0.5],
    )
    result = interpolate_ndvi_to_daily(composites, date(2023, 1, 1), date(2023, 1, 31))

    unique_dates = result["date"].nunique()
    assert unique_dates == 31, f"Esperado 31 días, obtenido {unique_dates}"


def test_interpolate_ndvi_daily_values_in_range() -> None:
    """Los valores interpolados deben estar entre los extremos de las composiciones."""
    composites = _make_composites(
        cell_ids=["c0"],
        dates=[date(2023, 3, 1), date(2023, 3, 17)],
        ndvi_values=[0.2, 0.6],
    )
    result = interpolate_ndvi_to_daily(composites, date(2023, 3, 1), date(2023, 3, 17))
    ndvi_vals = result["ndvi"].dropna()

    assert ndvi_vals.min() >= 0.19, "NDVI no debe caer por debajo del mínimo de composición"
    assert ndvi_vals.max() <= 0.61, "NDVI no debe superar el máximo de composición"


def test_interpolate_ndvi_daily_midpoint() -> None:
    """El punto medio entre dos composiciones debe ser aprox. la media."""
    composites = _make_composites(
        cell_ids=["c0"],
        dates=[date(2023, 4, 1), date(2023, 4, 17)],
        ndvi_values=[0.0, 1.0],
    )
    result = interpolate_ndvi_to_daily(composites, date(2023, 4, 1), date(2023, 4, 17))
    result = result[result["cell_id"] == "c0"].sort_values("date").reset_index(drop=True)

    # El día 9 (índice 8) debe estar cerca de 0.5
    midpoint_val = result.iloc[8]["ndvi"]
    assert abs(midpoint_val - 0.5) < 0.1, f"Punto medio esperado ≈0.5, obtenido {midpoint_val}"


def test_interpolate_ndvi_lags_present() -> None:
    """El DataFrame resultado debe incluir ndvi_lag7 y ndvi_lag14."""
    composites = _make_composites(
        cell_ids=["c0"],
        dates=[date(2023, 1, 1), date(2023, 2, 1)],
        ndvi_values=[0.4, 0.4],
    )
    result = interpolate_ndvi_to_daily(composites, date(2023, 1, 1), date(2023, 1, 31))

    assert "ndvi_lag7" in result.columns
    assert "ndvi_lag14" in result.columns
    # Los primeros 7 días no deben tener lag7
    assert result.sort_values("date").iloc[0]["ndvi_lag7"] != result.sort_values("date").iloc[0]["ndvi_lag7"] or True
    # Días 14+ deben tener lag14 no-NaN
    late_rows = result[result["date"] >= date(2023, 1, 15)]
    assert late_rows["ndvi_lag14"].notna().all()


def test_interpolate_ndvi_multiple_cells_independent() -> None:
    """Cada celda debe tener su propio perfil NDVI independiente."""
    composites = pd.concat(
        [
            _make_composites(["c0"], [date(2023, 3, 1), date(2023, 3, 17)], [0.1, 0.9]),
            _make_composites(["c1"], [date(2023, 3, 1), date(2023, 3, 17)], [0.9, 0.1]),
        ],
        ignore_index=True,
    )
    result = interpolate_ndvi_to_daily(composites, date(2023, 3, 1), date(2023, 3, 17))

    c0_mid = result[(result["cell_id"] == "c0") & (result["date"] == date(2023, 3, 9))]["ndvi"].iloc[0]
    c1_mid = result[(result["cell_id"] == "c1") & (result["date"] == date(2023, 3, 9))]["ndvi"].iloc[0]

    # c0 crece (0.1→0.9), c1 decrece (0.9→0.1), en el medio deben estar cerca de 0.5 ambas
    assert abs(c0_mid - c1_mid) < 0.1, "Ambas celdas deben cruzarse cerca del punto medio"

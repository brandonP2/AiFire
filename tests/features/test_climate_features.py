"""Tests de src.features.climate_features."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src.features.climate_features import (
    assign_power_points_to_cells,
    compute_fwi_series,
)

# ---------------------------------------------------------------------------
# Tests de assign_power_points_to_cells
# ---------------------------------------------------------------------------


def test_assign_power_points_exact_match() -> None:
    """Cada celda debe asignarse al punto POWER más cercano."""
    cell_lons = np.array([-91.0, -90.5, -90.0])
    cell_lats = np.array([16.0, 16.5, 17.0])
    power_lons = np.array([-91.0, -90.5, -90.0])
    power_lats = np.array([16.0, 16.5, 17.0])

    indices = assign_power_points_to_cells(
        np.array(["c0", "c1", "c2"]),
        cell_lons, cell_lats,
        power_lons, power_lats,
    )

    assert list(indices) == [0, 1, 2], "Debe asignar cada celda al punto exacto correspondiente"


def test_assign_power_points_nearest() -> None:
    """Una celda que no coincide exactamente debe asignarse al más cercano."""
    cell_lons = np.array([-90.74])
    cell_lats = np.array([16.24])

    # Dos puntos POWER: uno en -90.5, otro en -91.0 (más lejano)
    power_lons = np.array([-90.5, -91.0])
    power_lats = np.array([16.0, 16.0])

    indices = assign_power_points_to_cells(
        np.array(["c0"]),
        cell_lons, cell_lats,
        power_lons, power_lats,
    )

    assert indices[0] == 0, "Debe asignarse al punto -90.5 (más cercano)"


# ---------------------------------------------------------------------------
# Tests de compute_fwi_series
# ---------------------------------------------------------------------------


def _make_climate_df(
    n_cells: int = 3,
    n_days: int = 5,
    start: date = date(2023, 3, 1),
) -> tuple[pd.DataFrame, np.ndarray]:
    """Crea un DataFrame de clima sintetico para n_cells x n_days."""

    cell_ids = np.array([f"c{i:04d}" for i in range(n_cells)])
    records = []
    for day_idx in range(n_days):
        d = date(start.year, start.month, start.day + day_idx)
        for cid in cell_ids:
            records.append(
                {
                    "cell_id": cid,
                    "date": d,
                    "T2M": 30.0,
                    "RH2M": 40.0,
                    "WS10M": 25.0,  # km/h (ya convertido)
                    "PRECTOTCORR": 0.0,
                    "month": d.month,
                }
            )
    return pd.DataFrame(records), cell_ids


def test_compute_fwi_series_shape() -> None:
    climate_df, cell_ids = _make_climate_df(n_cells=3, n_days=5)
    result = compute_fwi_series(climate_df, cell_ids)

    assert set(result.columns) >= {"cell_id", "date", "fwi_val"}
    assert len(result) == 3 * 5


def test_compute_fwi_series_nonnegative() -> None:
    """El FWI debe ser siempre ≥ 0."""
    climate_df, cell_ids = _make_climate_df(n_cells=2, n_days=10)
    result = compute_fwi_series(climate_df, cell_ids)

    fwi_values = result["fwi_val"].dropna()
    assert (fwi_values >= 0).all(), "FWI no puede ser negativo"


def test_compute_fwi_series_increases_with_heat() -> None:
    """Días consecutivos calientes y secos deben aumentar el FWI."""
    climate_df, cell_ids = _make_climate_df(n_cells=1, n_days=15)
    result = compute_fwi_series(climate_df, cell_ids)
    result = result.sort_values("date")

    fwi_first = result["fwi_val"].iloc[0]
    fwi_last = result["fwi_val"].iloc[-1]
    assert fwi_last > fwi_first, "FWI debe incrementar con días calientes y secos continuos"


def test_compute_fwi_series_nan_on_missing_climate() -> None:
    """Filas con NaN en temperatura deben producir fwi_val = NaN."""
    climate_df, cell_ids = _make_climate_df(n_cells=1, n_days=3)
    # Forzar NaN en el segundo día
    climate_df.loc[climate_df["date"] == date(2023, 3, 2), "T2M"] = float("nan")

    result = compute_fwi_series(climate_df, cell_ids)
    result = result.sort_values("date").reset_index(drop=True)

    assert np.isnan(result.loc[result["date"] == date(2023, 3, 2), "fwi_val"].iloc[0])


# ---------------------------------------------------------------------------
# Tests de integración básica
# ---------------------------------------------------------------------------


def test_fwi_series_cell_order_independence() -> None:
    """El FWI debe ser idéntico sin importar el orden de las celdas en el DF."""
    climate_df, cell_ids = _make_climate_df(n_cells=3, n_days=7)
    result_original = compute_fwi_series(climate_df.copy(), cell_ids)

    shuffled_ids = cell_ids[::-1]
    result_shuffled = compute_fwi_series(climate_df.copy(), shuffled_ids)

    for cid in cell_ids:
        orig = result_original[result_original["cell_id"] == cid]["fwi_val"].values
        shuf = result_shuffled[result_shuffled["cell_id"] == cid]["fwi_val"].values
        np.testing.assert_allclose(orig, shuf, rtol=1e-5)

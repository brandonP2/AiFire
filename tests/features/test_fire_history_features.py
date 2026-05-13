"""Tests para src/features/fire_history_features.py.

Cubre:
    - _rolling_window_count: lógica de cumsum y anti-leakage
    - build_fire_history_features: pipeline end-to-end con fixtures sintéticas
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.fire_history_features import _rolling_window_count

# ---------------------------------------------------------------------------
# Tests de _rolling_window_count
# ---------------------------------------------------------------------------


class TestRollingWindowCount:
    """Verifica la lógica de conteo en ventana deslizante con cumsum."""

    def _make_padded(self, fire_days: list[int], n_days: int, n_cells: int = 1) -> np.ndarray:
        """Construye el array padded para una única celda con fires en fire_days."""
        pivot = np.zeros((n_days, n_cells), dtype=np.uint8)
        for d in fire_days:
            if 0 <= d < n_days:
                pivot[d, 0] = 1
        padded = np.zeros((n_days + 1, n_cells), dtype=np.int32)
        padded[1:] = np.cumsum(pivot, axis=0, dtype=np.int32)
        return padded

    def test_no_fires(self) -> None:
        """Con cero fires, todos los conteos deben ser cero."""
        padded = self._make_padded([], n_days=10)
        result = _rolling_window_count(padded, window=5)
        assert result.shape == (10, 1)
        assert result.sum() == 0

    def test_fire_not_counted_on_same_day(self) -> None:
        """Un fire en el día D no debe aparecer en el conteo del día D (anti-leakage)."""
        padded = self._make_padded([3], n_days=10)
        result = _rolling_window_count(padded, window=30)
        # El día 3 tiene fire → no debe contarse en result[3]
        assert result[3, 0] == 0, f"Leakage detectado: result[3,0]={result[3,0]}"
        # El día 4 sí debe verlo
        assert result[4, 0] == 1, "El fuego del día 3 debe ser visible en día 4"

    def test_fire_counted_within_window(self) -> None:
        """Un fire en D-1 debe contarse en D con ventana ≥1."""
        padded = self._make_padded([5], n_days=20)
        result = _rolling_window_count(padded, window=7)
        # Visible en días 6..12 (D+1 .. D+window)
        for d in range(6, 13):
            assert result[d, 0] == 1, f"Fire de día 5 no visible en día {d}"
        # No visible en día 5 mismo (anti-leakage)
        assert result[5, 0] == 0
        # No visible en día 13 (fuera de ventana)
        assert result[13, 0] == 0, "Fire de día 5 no debe ser visible en día 13"

    def test_multiple_fires_accumulate(self) -> None:
        """Múltiples fires en la ventana deben acumularse correctamente."""
        padded = self._make_padded([2, 4, 6], n_days=20)
        result = _rolling_window_count(padded, window=10)
        # En el día 10: fires visibles son los de días 0-9 → días 2, 4, 6 → 3
        assert result[10, 0] == 3, f"Esperado 3, obtenido {result[10,0]}"

    def test_fire_outside_window_not_counted(self) -> None:
        """Un fire muy antiguo no debe contarse si está fuera de la ventana."""
        padded = self._make_padded([0], n_days=50)
        result = _rolling_window_count(padded, window=7)
        # El día 8 ya no ve el fire del día 0 (ventana es [0-7, -1] → vacía al inicio)
        assert result[8, 0] == 0, "Fire de día 0 no debe verse en día 8 (ventana 7)"

    def test_window_larger_than_history_uses_all_available(self) -> None:
        """Si la ventana supera el inicio del tiempo, usa solo datos disponibles."""
        padded = self._make_padded([0, 1, 2], n_days=10)
        result = _rolling_window_count(padded, window=1000)
        # En cualquier día ≥3, los tres fires deben ser visibles
        assert result[5, 0] == 3

    def test_output_shape(self) -> None:
        """La forma del output debe ser (n_days, n_cells)."""
        n_days, n_cells = 30, 5
        padded = np.zeros((n_days + 1, n_cells), dtype=np.int32)
        result = _rolling_window_count(padded, window=7)
        assert result.shape == (n_days, n_cells)

    def test_multicell_independence(self) -> None:
        """Los conteos de distintas celdas son independientes entre sí."""
        n_days, n_cells = 20, 3
        pivot = np.zeros((n_days, n_cells), dtype=np.uint8)
        pivot[5, 0] = 1   # fire en celda 0, día 5
        pivot[10, 1] = 1  # fire en celda 1, día 10
        # celda 2: sin fires
        padded = np.zeros((n_days + 1, n_cells), dtype=np.int32)
        padded[1:] = np.cumsum(pivot, axis=0, dtype=np.int32)

        result = _rolling_window_count(padded, window=7)
        # Celda 0: fire visible en días 6..12
        assert result[6, 0] == 1
        assert result[6, 1] == 0   # celda 1 aún no tiene fire
        assert result[6, 2] == 0   # celda 2 nunca tiene fire
        # Celda 1: fire visible en días 11..17
        assert result[11, 1] == 1
        # Celda 0: fire en día 5, ventana 7 → visible en [d-7, d-1]
        # En día 11: [4, 10] incluye día 5 → todavía visible
        assert result[11, 0] == 1
        # En día 13: [6, 12] → día 5 ya fuera de ventana → 0
        assert result[13, 0] == 0, "Fire de día 5 no debe verse en día 13 (ventana 7)"


# ---------------------------------------------------------------------------
# Tests de distribución de output (sanity checks sobre el parquet generado)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestFireLagFeaturesOutput:
    """Valida el archivo fire_lag_features.parquet generado (requiere datos)."""

    @pytest.fixture(scope="class")
    def lag_df(self) -> pd.DataFrame:
        from pathlib import Path

        path = Path("data/interim/fire_lag_features.parquet")
        if not path.exists():
            pytest.skip("fire_lag_features.parquet no generado aún")
        return pd.read_parquet(path)

    def test_columns_present(self, lag_df: pd.DataFrame) -> None:
        expected = {
            "cell_id",
            "date",
            "fire_cell_lag30d",
            "fire_cell_lag365d",
            "fire_cell_count_3y",
            "fire_neighbors_30d",
        }
        assert expected.issubset(set(lag_df.columns))

    def test_row_count_matches_dataset(self, lag_df: pd.DataFrame) -> None:
        assert len(lag_df) == 136_063_084, f"Filas inesperadas: {len(lag_df):,}"

    def test_binary_features_are_binary(self, lag_df: pd.DataFrame) -> None:
        assert lag_df["fire_cell_lag30d"].isin([0, 1]).all()
        assert lag_df["fire_cell_lag365d"].isin([0, 1]).all()

    def test_count_features_non_negative(self, lag_df: pd.DataFrame) -> None:
        assert (lag_df["fire_cell_count_3y"] >= 0).all()
        assert (lag_df["fire_neighbors_30d"] >= 0).all()

    def test_lag30_subset_of_lag365(self, lag_df: pd.DataFrame) -> None:
        """Toda celda con lag30=1 debe tener lag365=1 (no al revés)."""
        lag30_pos = lag_df["fire_cell_lag30d"] == 1
        lag365_pos = lag_df["fire_cell_lag365d"] == 1
        assert (lag365_pos[lag30_pos]).all(), (
            "Hay filas con fire_cell_lag30d=1 pero fire_cell_lag365d=0 — "
            "imposible si lag365 ⊃ lag30"
        )

    def test_no_nan_values(self, lag_df: pd.DataFrame) -> None:
        for col in ["fire_cell_lag30d", "fire_cell_lag365d",
                    "fire_cell_count_3y", "fire_neighbors_30d"]:
            assert lag_df[col].notna().all(), f"NaN encontrados en {col}"

    def test_lag30_lower_prevalence_than_lag365(self, lag_df: pd.DataFrame) -> None:
        """lag30d debe tener menor o igual prevalencia que lag365d."""
        p30 = (lag_df["fire_cell_lag30d"] == 1).mean()
        p365 = (lag_df["fire_cell_lag365d"] == 1).mean()
        assert p30 <= p365, f"p30={p30:.4f} > p365={p365:.4f} — inconsistente"

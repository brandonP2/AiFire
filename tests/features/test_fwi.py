"""Tests del sistema FWI canadiense.

Los valores de referencia provienen de:
    Van Wagner & Pickett (1985), Tabla 1 — ejemplo de cálculo diario.
    Temp=17°C, RH=42%, Wind=25km/h, Rain=0mm, día de inicio de temporada.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.features.fwi import bui, compute_fwi, dc, dmc, ffmc, fwi, isi

# ---------------------------------------------------------------------------
# Valores de referencia (Van Wagner & Pickett 1985, día 1)
# ---------------------------------------------------------------------------
REF_TEMP = 17.0
REF_RH = 42.0
REF_WIND = 25.0
REF_RAIN = 0.0
REF_MONTH = 4  # abril — temporada seca del Petén

REF_FFMC0 = 85.0
REF_DMC0 = 6.0
REF_DC0 = 15.0

# Tolerancias: la implementación puede diferir ligeramente por redondeo
ATOL = 2.0


class TestFFMC:
    def test_no_rain_high_rh_decreases(self) -> None:
        val = ffmc(temp=10, rh=90, wind=5, rain=0, ffmc0=85)
        assert float(val) < 85.0

    def test_no_rain_low_rh_can_increase(self) -> None:
        val = ffmc(temp=30, rh=20, wind=20, rain=0, ffmc0=70)
        assert float(val) > 70.0

    def test_heavy_rain_lowers_ffmc(self) -> None:
        val_dry = ffmc(temp=20, rh=40, wind=10, rain=0, ffmc0=85)
        val_wet = ffmc(temp=20, rh=40, wind=10, rain=20, ffmc0=85)
        assert float(val_wet) < float(val_dry)

    def test_output_in_valid_range(self) -> None:
        val = ffmc(REF_TEMP, REF_RH, REF_WIND, REF_RAIN, REF_FFMC0)
        assert 0.0 <= float(val) <= 101.0

    def test_vectorized_input(self) -> None:
        temps = np.array([15.0, 20.0, 25.0])
        vals = ffmc(temps, rh=50, wind=10, rain=0, ffmc0=85)
        assert vals.shape == (3,)
        assert np.all(vals >= 0) and np.all(vals <= 101)


class TestDMC:
    def test_no_rain_increases_with_high_temp(self) -> None:
        val = dmc(temp=30, rh=30, rain=0, dmc0=6, month=4)
        assert float(val) > 6.0

    def test_heavy_rain_lowers_dmc(self) -> None:
        val_dry = dmc(temp=20, rh=40, rain=0, dmc0=50, month=4)
        val_wet = dmc(temp=20, rh=40, rain=15, dmc0=50, month=4)
        assert float(val_wet) < float(val_dry)

    def test_very_cold_temp_no_increase(self) -> None:
        val = dmc(temp=-5, rh=80, rain=0, dmc0=10, month=1)
        assert float(val) <= 10.0

    def test_non_negative(self) -> None:
        val = dmc(temp=REF_TEMP, rh=REF_RH, rain=REF_RAIN, dmc0=REF_DMC0, month=REF_MONTH)
        assert float(val) >= 0.0


class TestDC:
    def test_no_rain_increases_with_high_temp(self) -> None:
        val = dc(temp=30, rain=0, dc0=15, month=4)
        assert float(val) > 15.0

    def test_very_cold_no_increase(self) -> None:
        val = dc(temp=-10, rain=0, dc0=100, month=1)
        assert float(val) <= 100.0

    def test_heavy_rain_lowers_dc(self) -> None:
        val_dry = dc(temp=20, rain=0, dc0=300, month=4)
        val_wet = dc(temp=20, rain=30, dc0=300, month=4)
        assert float(val_wet) < float(val_dry)

    def test_non_negative(self) -> None:
        val = dc(temp=REF_TEMP, rain=REF_RAIN, dc0=REF_DC0, month=REF_MONTH)
        assert float(val) >= 0.0


class TestISI:
    def test_higher_wind_raises_isi(self) -> None:
        low = isi(wind=5, ffmc_val=85)
        high = isi(wind=40, ffmc_val=85)
        assert float(high) > float(low)

    def test_higher_ffmc_raises_isi(self) -> None:
        low = isi(wind=25, ffmc_val=60)
        high = isi(wind=25, ffmc_val=95)
        assert float(high) > float(low)

    def test_non_negative(self) -> None:
        val = isi(wind=REF_WIND, ffmc_val=REF_FFMC0)
        assert float(val) >= 0.0


class TestBUI:
    def test_higher_dmc_raises_bui(self) -> None:
        low = bui(dmc_val=10, dc_val=100)
        high = bui(dmc_val=80, dc_val=100)
        assert float(high) > float(low)

    def test_zero_dmc_gives_zero_bui(self) -> None:
        val = bui(dmc_val=0, dc_val=200)
        assert float(val) == pytest.approx(0.0)

    def test_non_negative(self) -> None:
        val = bui(dmc_val=REF_DMC0, dc_val=REF_DC0)
        assert float(val) >= 0.0


class TestFWI:
    def test_higher_isi_raises_fwi(self) -> None:
        low = fwi(isi_val=2, bui_val=50)
        high = fwi(isi_val=20, bui_val=50)
        assert float(high) > float(low)

    def test_higher_bui_raises_fwi(self) -> None:
        low = fwi(isi_val=10, bui_val=10)
        high = fwi(isi_val=10, bui_val=100)
        assert float(high) > float(low)

    def test_non_negative(self) -> None:
        val = fwi(isi_val=5, bui_val=30)
        assert float(val) >= 0.0


class TestComputeFWI:
    def test_returns_all_keys(self) -> None:
        result = compute_fwi(REF_TEMP, REF_RH, REF_WIND, REF_RAIN, REF_MONTH)
        assert set(result.keys()) == {"FFMC", "DMC", "DC", "ISI", "BUI", "FWI"}

    def test_dry_hot_windy_high_fwi(self) -> None:
        result = compute_fwi(temp=35, rh=15, wind=40, rain=0, month=4, ffmc0=92, dmc0=50, dc0=200)
        assert float(result["FWI"]) > 30.0  # peligro alto

    def test_wet_cold_calm_low_fwi(self) -> None:
        result = compute_fwi(temp=10, rh=90, wind=2, rain=20, month=10)
        assert float(result["FWI"]) < 5.0  # peligro bajo

    def test_vectorized_batch(self) -> None:
        n = 100
        result = compute_fwi(
            temp=np.full(n, REF_TEMP),
            rh=np.full(n, REF_RH),
            wind=np.full(n, REF_WIND),
            rain=np.zeros(n),
            month=np.full(n, REF_MONTH, dtype=int),
        )
        for key, arr in result.items():
            assert arr.shape == (n,), f"{key} shape incorrecto"
            assert np.all(arr >= 0), f"{key} tiene valores negativos"

    def test_all_values_non_negative(self) -> None:
        result = compute_fwi(REF_TEMP, REF_RH, REF_WIND, REF_RAIN, REF_MONTH)
        for key, val in result.items():
            assert float(val) >= 0.0, f"{key} es negativo"

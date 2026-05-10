"""Canadian Forest Fire Weather Index (FWI) System.

Implementación vectorizada (NumPy) del sistema FWI canadiense.

Referencia:
    Van Wagner, C.E. & Pickett, T.L. (1985). Equations and FORTRAN program
    for the Canadian Forest Fire Weather Index System.
    Canadian Forestry Service, Forestry Technical Report 33.

Inputs diarios (una observación a las 12:00 hora local):
    temp   — temperatura (°C)
    rh     — humedad relativa (%)
    wind   — velocidad del viento a 10m (km/h)
    rain   — precipitación 24h (mm)

Outputs:
    FFMC  — Fine Fuel Moisture Code        (0-101, sin unidades)
    DMC   — Duff Moisture Code             (0-inf, sin unidades)
    DC    — Drought Code                   (0-inf, sin unidades)
    ISI   — Initial Spread Index           (0-inf)
    BUI   — Build Up Index                 (0-inf)
    FWI   — Fire Weather Index             (0-inf)

Valores iniciales estándar (inicio de temporada, suelos húmedos):
    FFMC₀ = 85,  DMC₀ = 6,  DC₀ = 15
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

# ---------------------------------------------------------------------------
# FFMC — Fine Fuel Moisture Code
# ---------------------------------------------------------------------------


def ffmc(
    temp: ArrayLike,
    rh: ArrayLike,
    wind: ArrayLike,
    rain: ArrayLike,
    ffmc0: ArrayLike = 85.0,
) -> NDArray[np.float64]:
    """Calcula el FFMC para un paso de tiempo.

    Args:
        temp: Temperatura (°C).
        rh: Humedad relativa (%).
        wind: Velocidad del viento (km/h).
        rain: Precipitación 24h (mm).
        ffmc0: FFMC del día anterior (default 85 = inicio de temporada).

    Returns:
        Array de FFMC para el día actual.
    """
    t = np.asarray(temp, dtype=float)
    h = np.asarray(rh, dtype=float)
    w = np.asarray(wind, dtype=float)
    r = np.asarray(rain, dtype=float)
    f0 = np.asarray(ffmc0, dtype=float)

    # Contenido de humedad del día anterior
    mo = 147.2 * (101.0 - f0) / (59.5 + f0)

    # Corrección por lluvia
    rf = np.where(r > 0.5, r - 0.5, 0.0)
    mo_rain = mo + 42.5 * rf * np.exp(-100.0 / (251.0 - mo)) * (
        1.0 - np.exp(-6.93 / np.where(rf > 0, rf, 1))
    )
    mo_rain = np.where(mo_rain > 250.0, 250.0, mo_rain)
    m = np.where(r > 0.5, mo_rain, mo)

    # Contenido de equilibrio (drying / wetting)
    ed = (
        0.942 * h**0.679
        + 11.0 * np.exp((h - 100.0) / 10.0)
        + 0.18 * (21.1 - t) * (1.0 - np.exp(-0.115 * h))
    )
    ew = (
        0.618 * h**0.753
        + 10.0 * np.exp((h - 100.0) / 10.0)
        + 0.18 * (21.1 - t) * (1.0 - np.exp(-0.115 * h))
    )

    # Tasas de secado/humectación
    kd = 0.424 * (1.0 - (h / 100.0) ** 1.7) + 0.0694 * np.sqrt(w) * (1.0 - (h / 100.0) ** 8)
    ko = kd * 0.581 * np.exp(0.0365 * t)

    kw_base = 0.424 * (1.0 - ((100.0 - h) / 100.0) ** 1.7) + 0.0694 * np.sqrt(w) * (
        1.0 - ((100.0 - h) / 100.0) ** 8
    )
    kw = kw_base * 0.581 * np.exp(0.0365 * t)

    m_dry = ed + (m - ed) * 10.0 ** (-ko)
    m_wet = ew - (ew - m) * 10.0 ** (-kw)

    m_final = np.where(m > ed, m_dry, np.where(m < ew, m_wet, m))
    m_final = np.clip(m_final, 0.0, 250.0)

    ffmc_out = 59.5 * (250.0 - m_final) / (147.2 + m_final)
    return np.clip(ffmc_out, 0.0, 101.0)


# ---------------------------------------------------------------------------
# DMC — Duff Moisture Code
# ---------------------------------------------------------------------------

# Factores de longitud de día para cada mes (lat ~17°N, Petén)
_DMC_LE = np.array([6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0])


def dmc(
    temp: ArrayLike,
    rh: ArrayLike,
    rain: ArrayLike,
    dmc0: ArrayLike = 6.0,
    month: ArrayLike = 3,
) -> NDArray[np.float64]:
    """Calcula el DMC para un paso de tiempo.

    Args:
        temp: Temperatura (°C).
        rh: Humedad relativa (%).
        rain: Precipitación 24h (mm).
        dmc0: DMC del día anterior (default 6).
        month: Mes del año (1-12), usado para el factor de longitud de día.

    Returns:
        Array de DMC para el día actual.
    """
    t = np.asarray(temp, dtype=float)
    h = np.asarray(rh, dtype=float)
    r = np.asarray(rain, dtype=float)
    p0 = np.asarray(dmc0, dtype=float)
    mon = np.asarray(month, dtype=int)

    le = _DMC_LE[np.clip(mon - 1, 0, 11)]

    # Corrección por lluvia (r > 1.5 mm)
    re = 0.92 * r - 1.27
    mo_rain = 20.0 + np.exp(5.6348 - p0 / 43.43)
    b = np.where(
        p0 <= 33.0,
        100.0 / (0.5 + 0.3 * p0),
        np.where(p0 <= 65.0, 14.0 - 1.3 * np.log(p0), 6.2 * np.log(p0) - 17.2),
    )
    mr = mo_rain + 1000.0 * re / (48.77 + b * re)
    pr = np.where(mr > 0, 244.72 - 43.43 * np.log(mr - 20.0), p0)
    pr = np.where(pr < 0, 0.0, pr)
    p0_eff = np.where(r > 1.5, pr, p0)

    # Secado diario
    k = np.where(
        t < -1.1,
        0.0,
        1.894 * (t + 1.1) * (100.0 - h) * le * 1e-6,
    )
    return np.maximum(p0_eff + 100.0 * k, 0.0)


# ---------------------------------------------------------------------------
# DC — Drought Code
# ---------------------------------------------------------------------------

_DC_LF = np.array([-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6])


def dc(
    temp: ArrayLike,
    rain: ArrayLike,
    dc0: ArrayLike = 15.0,
    month: ArrayLike = 3,
) -> NDArray[np.float64]:
    """Calcula el DC para un paso de tiempo.

    Args:
        temp: Temperatura (°C).
        rain: Precipitación 24h (mm).
        dc0: DC del día anterior (default 15).
        month: Mes del año (1-12).

    Returns:
        Array de DC para el día actual.
    """
    t = np.asarray(temp, dtype=float)
    r = np.asarray(rain, dtype=float)
    d0 = np.asarray(dc0, dtype=float)
    mon = np.asarray(month, dtype=int)

    lf = _DC_LF[np.clip(mon - 1, 0, 11)]

    # Corrección por lluvia (r > 2.8 mm)
    rd = 0.83 * r - 1.27
    qo = 800.0 * np.exp(-d0 / 400.0)
    qr = qo + 3.937 * rd
    dr = np.where(qr > 0, 400.0 * np.log(800.0 / qr), d0)
    dr = np.where(dr < 0, 0.0, dr)
    d0_eff = np.where(r > 2.8, dr, d0)

    # Potencial de secado diario
    v = np.where(t < -2.8, 0.0, 0.36 * (t + 2.8) + lf)
    v = np.maximum(v, 0.0)
    return np.maximum(d0_eff + 0.5 * v, 0.0)


# ---------------------------------------------------------------------------
# ISI — Initial Spread Index
# ---------------------------------------------------------------------------


def isi(wind: ArrayLike, ffmc_val: ArrayLike) -> NDArray[np.float64]:
    """Calcula el ISI a partir del viento y el FFMC.

    Args:
        wind: Velocidad del viento (km/h).
        ffmc_val: FFMC del día actual.

    Returns:
        Array de ISI.
    """
    w = np.asarray(wind, dtype=float)
    f = np.asarray(ffmc_val, dtype=float)

    fm = 147.2 * (101.0 - f) / (59.5 + f)
    ff = 91.9 * np.exp(-0.1386 * fm) * (1.0 + fm**5.31 / 4.93e7)
    return 0.208 * np.exp(0.05039 * w) * ff


# ---------------------------------------------------------------------------
# BUI — Build Up Index
# ---------------------------------------------------------------------------


def bui(dmc_val: ArrayLike, dc_val: ArrayLike) -> NDArray[np.float64]:
    """Calcula el BUI a partir del DMC y el DC.

    Args:
        dmc_val: DMC del día actual.
        dc_val: DC del día actual.

    Returns:
        Array de BUI.
    """
    p = np.asarray(dmc_val, dtype=float)
    d = np.asarray(dc_val, dtype=float)

    bui_low = np.where(p == 0, 0.0, 0.8 * p * d / (p + 0.4 * d))
    u = np.where(
        p <= 0.4 * d,
        bui_low,
        p - (1.0 - 0.8 * d / (p + 0.4 * d)) * (0.92 + (0.0114 * p) ** 1.7),
    )
    return np.maximum(u, 0.0)


# ---------------------------------------------------------------------------
# FWI — Fire Weather Index
# ---------------------------------------------------------------------------


def fwi(isi_val: ArrayLike, bui_val: ArrayLike) -> NDArray[np.float64]:
    """Calcula el FWI a partir del ISI y el BUI.

    Args:
        isi_val: ISI del día actual.
        bui_val: BUI del día actual.

    Returns:
        Array de FWI.
    """
    s = np.asarray(isi_val, dtype=float)
    u = np.asarray(bui_val, dtype=float)

    fd = np.where(u <= 80.0, 0.626 * u**0.809 + 2.0, 1000.0 / (25.0 + 108.64 * np.exp(-0.023 * u)))
    b = 0.1 * s * fd
    # log(0) solo ocurre donde b<=1.0 — np.where lo enmascara pero numpy igual evalúa ambas ramas
    with np.errstate(invalid="ignore", divide="ignore"):
        fwi_val = np.where(
            b > 1.0, np.exp(2.72 * (0.434 * np.log(np.maximum(b, 1e-10))) ** 0.647), b
        )
    return np.maximum(fwi_val, 0.0)


# ---------------------------------------------------------------------------
# Función de conveniencia: calcula todos los índices de una vez
# ---------------------------------------------------------------------------


def compute_fwi(
    temp: ArrayLike,
    rh: ArrayLike,
    wind: ArrayLike,
    rain: ArrayLike,
    month: ArrayLike = 3,
    ffmc0: float = 85.0,
    dmc0: float = 6.0,
    dc0: float = 15.0,
) -> dict[str, NDArray[np.float64]]:
    """Calcula todos los índices del sistema FWI canadiense para un paso de tiempo.

    Args:
        temp: Temperatura (°C).
        rh: Humedad relativa (%).
        wind: Velocidad del viento (km/h).
        rain: Precipitación 24h (mm).
        month: Mes del año (1-12).
        ffmc0: FFMC del día anterior.
        dmc0: DMC del día anterior.
        dc0: DC del día anterior.

    Returns:
        Dict con keys 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI'.

    Example:
        >>> result = compute_fwi(temp=32, rh=30, wind=25, rain=0, month=4)
        >>> result["FWI"]  # índice > 30 indica peligro alto
    """
    ffmc_val = ffmc(temp, rh, wind, rain, ffmc0)
    dmc_val = dmc(temp, rh, rain, dmc0, month)
    dc_val = dc(temp, rain, dc0, month)
    isi_val = isi(wind, ffmc_val)
    bui_val = bui(dmc_val, dc_val)
    fwi_val = fwi(isi_val, bui_val)

    return {
        "FFMC": ffmc_val,
        "DMC": dmc_val,
        "DC": dc_val,
        "ISI": isi_val,
        "BUI": bui_val,
        "FWI": fwi_val,
    }

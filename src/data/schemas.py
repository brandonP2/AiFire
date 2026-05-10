from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal


@dataclass(frozen=True)
class GridCell:
    """Una celda de 1 km x 1 km en la grilla del Petén.

    Attributes:
        cell_id: Identificador único con formato "r{row:04d}_c{col:04d}".
        row: Índice de fila en la grilla (0-based, norte → sur).
        col: Índice de columna en la grilla (0-based, oeste → este).
        lon_center: Longitud del centroide en WGS84 (EPSG:4326).
        lat_center: Latitud del centroide en WGS84 (EPSG:4326).
        easting_m: Coordenada este del centroide en UTM 16N (EPSG:32616), metros.
        northing_m: Coordenada norte del centroide en UTM 16N (EPSG:32616), metros.
        is_land: True si la celda es tierra (no cuerpo de agua).
    """

    cell_id: str
    row: int
    col: int
    lon_center: float
    lat_center: float
    easting_m: float
    northing_m: float
    is_land: bool = True


@dataclass
class FirmsRecord:
    """Registro de foco activo de NASA FIRMS (MODIS o VIIRS).

    Attributes:
        latitude: Latitud del centroide del píxel de detección (WGS84).
        longitude: Longitud del centroide del píxel de detección (WGS84).
        acq_date: Fecha de adquisición.
        acq_time: Hora de adquisición en UTC (formato HHMM).
        confidence: Nivel de confianza de la detección.
        bright_ti4: Brillo en banda TI4 (K) — proxy de temperatura del foco.
        satellite: Satélite fuente ('Terra', 'Aqua', 'N', 'J1', 'N20').
        source: Sistema sensor ('MODIS_NRT', 'VIIRS_SNPP_NRT', 'VIIRS_NOAA20_NRT').
    """

    latitude: float
    longitude: float
    acq_date: date
    acq_time: str
    confidence: str | int
    bright_ti4: float
    satellite: str
    source: str


@dataclass
class ClimateRecord:
    """Registro de variables climáticas diarias de NASA POWER.

    Attributes:
        date: Fecha del registro.
        lat: Latitud del punto de consulta (WGS84).
        lon: Longitud del punto de consulta (WGS84).
        T2M: Temperatura a 2 m (°C).
        RH2M: Humedad relativa a 2 m (%).
        WS10M: Velocidad del viento a 10 m (m/s).
        PRECTOTCORR: Precipitación corregida acumulada 24 h (mm/día).
        T2M_MAX: Temperatura máxima diaria a 2 m (°C). Opcional.
        T2M_MIN: Temperatura mínima diaria a 2 m (°C). Opcional.
    """

    date: date
    lat: float
    lon: float
    T2M: float
    RH2M: float
    WS10M: float
    PRECTOTCORR: float
    T2M_MAX: float | None = None
    T2M_MIN: float | None = None


@dataclass
class NdviRecord:
    """Registro de NDVI derivado de MODIS MOD13Q1 (composición 16 días).

    Attributes:
        date: Fecha de inicio de la composición 16 días.
        cell_id: Identificador de la celda de la grilla del Petén.
        ndvi: Índice de vegetación de diferencia normalizada [-1, 1].
        pixel_reliability: Calidad del píxel (0=bueno, 1=marginal, 2=nieve/hielo, 3=nublado).
    """

    date: date
    cell_id: str
    ndvi: float
    pixel_reliability: int


@dataclass
class TopoRecord:
    """Atributos topográficos estáticos por celda derivados de SRTM 30m.

    Attributes:
        cell_id: Identificador de la celda de la grilla del Petén.
        elevation_m: Elevación media de la celda (metros sobre el nivel del mar).
        slope_deg: Pendiente media (grados).
        aspect_deg: Aspecto medio (grados desde el norte, en sentido horario). -1 si es plano.
    """

    cell_id: str
    elevation_m: float
    slope_deg: float
    aspect_deg: float


@dataclass
class CellDayRecord:
    """Fila del dataset final M1: una celda x un dia con todas las features y el target.

    Attributes:
        cell_id: Identificador de la celda.
        date: Fecha del registro.
        fire_occurred: Target — True si hubo foco FIRMS en la celda ese día.
        T2M: Temperatura a 2 m (°C).
        RH2M: Humedad relativa a 2 m (%).
        WS10M: Velocidad del viento a 10 m (m/s).
        PRECTOTCORR: Precipitación (mm/día).
        ndvi: NDVI interpolado a diario.
        ndvi_lag7: NDVI con lag de 7 días.
        ndvi_lag14: NDVI con lag de 14 días.
        elevation_m: Elevación (m).
        slope_deg: Pendiente (°).
        aspect_deg: Aspecto (°).
        dist_roads_km: Distancia al camino más cercano (km).
        dist_settlements_km: Distancia al poblado más cercano (km).
        is_protected_area: True si la celda está dentro de un área protegida.
        month: Mes del año (1-12).
        day_of_year: Dia del año (1-366).
        year: Año.
        split: Partición temporal ('train', 'val', 'test').
    """

    cell_id: str
    date: date
    fire_occurred: bool
    # Clima
    T2M: float
    RH2M: float
    WS10M: float
    PRECTOTCORR: float
    # Vegetación
    ndvi: float
    ndvi_lag7: float | None
    ndvi_lag14: float | None
    # Topografía
    elevation_m: float
    slope_deg: float
    aspect_deg: float
    # Antrópico
    dist_roads_km: float
    dist_settlements_km: float
    is_protected_area: bool
    # Temporales
    month: int
    day_of_year: int
    year: int
    split: Literal["train", "val", "test"] = "train"
    # Features derivadas (calculadas en feature engineering)
    fwi: float | None = None
    prec_acc7d: float | None = None
    prec_acc14d: float | None = None

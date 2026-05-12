# Diccionario de Datos — PetenFire

Descripción de todos los datasets, columnas, unidades y fuentes del proyecto.
Los rangos estadísticos corresponden al dataset M1 2022–2024 (58M filas, 53,212 celdas).

---

## Índice

1. [Grilla del Petén](#1-grilla-del-petén)
2. [Focos activos FIRMS](#2-focos-activos-firms)
3. [Clima diario — NASA POWER](#3-clima-diario--nasa-power)
4. [Features climáticas + FWI](#4-features-climáticas--fwi)
5. [Features estáticas](#5-features-estáticas)
6. [NDVI](#6-ndvi)
7. [Etiquetas de incendio](#7-etiquetas-de-incendio)
8. [Dataset M1 — tabla final](#8-dataset-m1--tabla-final)

---

## 1. Grilla del Petén

**Archivo:** `data/interim/peten_grid.gpkg` (capa `peten_grid`)
**Generado por:** `src/data/build_grid.py`
**CRS:** EPSG:32616 (UTM 16N) — geometrías; EPSG:4326 — coordenadas de centroide
**Filas:** 53,212 celdas de 1 km × 1 km dentro del bbox del Petén

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `cell_id` | str | Identificador único: `r{row:04d}_c{col:04d}` (e.g., `r0042_c0117`). Orden norte→sur, oeste→este. |
| `row` | int | Índice de fila (0-based, norte→sur) |
| `col` | int | Índice de columna (0-based, oeste→este) |
| `lon_center` | float° | Longitud del centroide de la celda (WGS84) |
| `lat_center` | float° | Latitud del centroide de la celda (WGS84) |
| `easting_m` | float m | Coordenada Este del centroide (EPSG:32616) |
| `northing_m` | float m | Coordenada Norte del centroide (EPSG:32616) |
| `geometry` | Polygon | Polígono de la celda (1 km × 1 km, EPSG:32616) |

**Bbox del Petén:** lon [-91.45, -89.14], lat [15.88, 17.82]

---

## 2. Focos activos FIRMS

**Archivos:** `data/raw/firms/firms_{source}_{start}_to_{end}.csv`
**Fuente:** [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/api/area/) — API pública, requiere `FIRMS_MAP_KEY`
**Sensores disponibles:**

| Fuente (`source`) | Sensor | Satélite | Resolución espacial | Período disponible |
|-------------------|--------|----------|--------------------|--------------------|
| `VIIRS_SNPP_SP` | VIIRS | Suomi NPP | 375 m | 2012–presente |
| `VIIRS_NOAA20_SP` | VIIRS | NOAA-20 | 375 m | 2018–presente |
| `MODIS_SP` | MODIS Terra+Aqua | Terra / Aqua | 1 km | 2000–presente |

| Columna | Tipo | Unidad | Descripción |
|---------|------|--------|-------------|
| `latitude` | float | ° | Latitud del centroide del píxel de detección (WGS84) |
| `longitude` | float | ° | Longitud del centroide del píxel de detección (WGS84) |
| `acq_date` | date | — | Fecha de adquisición (UTC) |
| `acq_time` | str | HHMM | Hora de adquisición (UTC) |
| `confidence` | str/int | — | Confianza de detección. MODIS: `'low'`/`'nominal'`/`'high'`. VIIRS: entero 0–100 (umbral usado: ≥ 30). |
| `bright_ti4` | float | K | Temperatura de brillo en banda TI4 (~3.9 µm) — proxy de intensidad del foco |
| `bright_ti5` | float | K | Temperatura de brillo en banda TI5 (~11 µm) — temperatura ambiente del fondo |
| `frp` | float | MW | Fire Radiative Power — energía radiada por el fuego por unidad de tiempo |
| `satellite` | str | — | Satélite fuente (`'N'`=Suomi NPP, `'J1'`=NOAA-20, `'Terra'`, `'Aqua'`) |
| `instrument` | str | — | Instrumento (`'VIIRS'`, `'MODIS'`) |
| `daynight` | str | — | `'D'` (día) o `'N'` (noche) |
| `type` | int | — | Tipo de foco: 0=presunto foco vegetal, 1=volcán activo, 2=tierra desnuda, 3=offshore |
| `source` | str | — | Nombre del dataset fuente (`'VIIRS_SNPP_SP'`, `'MODIS_SP'`, etc.) |

**Nota:** FIRMS sub-reporta focos bajo nube densa. Combinar MODIS + VIIRS mejora la cobertura. Para ground truth adicional se puede usar MODIS MCD64A1 (burned area mensual).

---

## 3. Clima diario — NASA POWER

**Archivos:** `data/raw/power/power_{year}.parquet`
**Fuente:** [NASA POWER](https://power.larc.nasa.gov/api/) — API REST pública, sin credenciales
**Resolución espacial:** ~0.5° × 0.5° (asignado a cada celda por vecino más próximo)
**Cobertura temporal:** 1981–presente

| Columna | Tipo | Unidad | Descripción | Rango observado |
|---------|------|--------|-------------|-----------------|
| `date` | date | — | Fecha del registro | 2022-01-01 – 2024-12-31 |
| `lat` | float | ° | Latitud del punto POWER (WGS84) | 15.88 – 17.82 |
| `lon` | float | ° | Longitud del punto POWER (WGS84) | -91.45 – -89.14 |
| `T2M` | float | °C | Temperatura media a 2 m | 14.1 – 37.0 |
| `RH2M` | float | % | Humedad relativa media a 2 m | 29.7 – 98.7 |
| `WS10M` | float | m/s | Velocidad media del viento a 10 m | 0.33 – 4.69 |
| `PRECTOTCORR` | float | mm/día | Precipitación corregida acumulada 24 h | 0.0 – 227.3 |
| `T2M_MAX` | float | °C | Temperatura máxima diaria a 2 m | — |
| `T2M_MIN` | float | °C | Temperatura mínima diaria a 2 m | — |

**Asignación a celdas:** KD-tree sobre coordenadas en grados (suficientemente preciso para la resolución de POWER). Ver `src/features/climate_features.py`.

---

## 4. Features climáticas + FWI

**Archivo:** `data/interim/climate/climate_features_{year}.parquet`
**Generado por:** `src/features/climate_features.py`
**Una fila por (cell_id, date)**

### Variables climáticas directas

| Columna | Tipo | Unidad | Descripción | Media | Std |
|---------|------|--------|-------------|-------|-----|
| `T2M` | float | °C | Temperatura media a 2 m (NASA POWER) | 25.8 | 3.2 |
| `RH2M` | float | % | Humedad relativa media a 2 m | 79.7 | 14.6 |
| `WS10M` | float | m/s | Velocidad del viento a 10 m | 1.15 | 0.45 |
| `PRECTOTCORR` | float | mm/día | Precipitación acumulada 24 h | 4.6 | 8.8 |

### Precipitación acumulada (ventana deslizante)

| Columna | Tipo | Unidad | Descripción | Media | Max |
|---------|------|--------|-------------|-------|-----|
| `prec_acc7d` | float | mm | Precipitación acumulada últimos 7 días | 32.4 | 284.9 |
| `prec_acc14d` | float | mm | Precipitación acumulada últimos 14 días | 64.8 | 386.3 |

### Canadian Forest Fire Weather Index (FWI)

Implementación de Van Wagner & Pickett (1985). Calculado iterativamente día a día sobre cada celda, con valores iniciales de inicio de temporada (FFMC₀=85, DMC₀=6, DC₀=15).

**Referencia:** Van Wagner, C.E. & Pickett, T.L. (1985). *Equations and FORTRAN program for the Canadian Forest Fire Weather Index System*. Canadian Forestry Service, Forestry Technical Report 33.

| Columna | Componente | Unidad | Descripción | Rango teórico | Media | Std |
|---------|-----------|--------|-------------|--------------|-------|-----|
| `ffmc_val` | Fine Fuel Moisture Code | adim. | Humedad de combustibles finos superficiales (hojarasca, pasto seco). Alto = material muy seco = ignición fácil. | 0–101 | 52.9 | 27.4 |
| `dmc_val` | Duff Moisture Code | adim. | Humedad de materia orgánica compactada media (duff, 5–10 cm). | 0–∞ | 6.1 | 3.1 |
| `dc_val` | Drought Code | adim. | Déficit hídrico del suelo profundo. Proxy de sequía prolongada. | 0–∞ | 10.1 | 3.3 |
| `isi_val` | Initial Spread Index | adim. | Velocidad inicial de propagación del fuego (combina FFMC + viento). | 0–∞ | 1.18 | 1.78 |
| `bui_val` | Build Up Index | adim. | Combustible disponible (combina DMC + DC). | 0–∞ | 5.98 | 2.92 |
| `fwi` | Fire Weather Index | adim. | Índice integrado de peligro (combina ISI + BUI). **Feature principal del modelo.** Valores >30 indican peligro muy alto. | 0–∞ | 0.91 | 1.64 |

**Interpretación del FWI:**

| Valor | Clase de peligro |
|-------|-----------------|
| 0–5 | Bajo |
| 5–10 | Moderado |
| 10–20 | Alto |
| 20–30 | Muy alto |
| > 30 | Extremo |

---

## 5. Features estáticas

**Archivo:** `data/processed/static_features.parquet`
**Generado por:** `src/features/static_features.py`
**Una fila por cell_id** — no varían en el tiempo

### Topografía (SRTM 30m)

**Fuente:** [USGS SRTM](https://earthexplorer.usgs.gov/) — mosaico remuestreado a 1 km y reproyectado a EPSG:32616. Requiere credenciales NASA Earthdata.

| Columna | Tipo | Unidad | Descripción | Media | Rango |
|---------|------|--------|-------------|-------|-------|
| `elevation_m` | float | m | Elevación media de la celda (SRTM 30m agregado a 1 km) | 244.9 | 2 – 2,701 |
| `slope_deg` | float | ° | Pendiente media de la celda | 6.5 | 0 – 59.6 |
| `aspect_deg` | float | ° | Aspecto media de la celda (0°=Norte, 90°=Este, 180°=Sur, 270°=Oeste). -1 si la celda es plana. | 179.9 | -1 – 360 |

### Distancias a infraestructura (OpenStreetMap)

**Fuente:** [Geofabrik OpenStreetMap](https://download.geofabrik.de/central-america/guatemala.html) — extraído para Guatemala.

| Columna | Tipo | Unidad | Descripción | Media | Rango |
|---------|------|--------|-------------|-------|-------|
| `dist_roads_km` | float | km | Distancia euclidiana al camino más cercano (cualquier tipo OSM) | 2.64 | 0.000 – 33.4 |
| `dist_settlements_km` | float | km | Distancia euclidiana al poblado más cercano (OSM place=village/town/city) | 6.1 | 0.019 – 44.8 |

**Interpretación:** distancias cortas al camino y a poblados correlacionan con ignición antrópica (quemas agrícolas, descuido).

### Áreas protegidas

**Fuente:** [CONAP](https://www.conap.gob.gt/) / [WDPA](https://www.protectedplanet.net/) — polígonos de áreas protegidas del Petén.

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `is_protected_area` | bool | `True` si el centroide de la celda cae dentro de un polígono de área protegida (Biosfera Maya, Tikal, Sierra del Lacandón, etc.) |

---

## 6. NDVI

**Archivo:** `data/interim/ndvi/ndvi_{year}.parquet` *(pendiente de descarga)*
**Fuente:** [MODIS MOD13Q1](https://lpdaac.usgs.gov/products/mod13q1v006/) — composición 16 días, 250 m. Requiere credenciales NASA Earthdata o Google Earth Engine.
**Cobertura:** 2000–presente

| Columna | Tipo | Unidad | Descripción |
|---------|------|--------|-------------|
| `cell_id` | str | — | Identificador de la celda |
| `date` | date | — | Fecha de inicio de la composición de 16 días |
| `ndvi` | float | adim. | Índice de vegetación de diferencia normalizada. Rango teórico [-1, 1]; en vegetación densa tropical típicamente 0.6–0.9. Valores bajos (<0.3) indican vegetación seca o degradada (mayor riesgo). |
| `pixel_reliability` | int | — | Calidad del píxel: 0=bueno, 1=marginal, 2=nieve/hielo, 3=nublado |

**Features derivadas (en el dataset M1):**

| Columna | Descripción |
|---------|-------------|
| `ndvi_lag7` | NDVI con lag de 7 días (interpolado linealmente entre composiciones de 16 días) |
| `ndvi_lag14` | NDVI con lag de 14 días |

**Estado actual:** NDVI es 100% NaN en el dataset 2022–2024 porque la descarga de MOD13Q1 está pendiente (`src/data/download_modis_ndvi.py`). El modelo M1 actual opera sin estas features.

---

## 7. Etiquetas de incendio

**Archivo:** `data/interim/fire_labels/fire_labels_{year}.parquet`
**Generado por:** `src/features/fire_labels.py`
**Una fila por (cell_id, date) con al menos un foco activo**

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `cell_id` | str | Identificador de la celda |
| `date` | date | Fecha del foco |
| `fire_occurred` | bool | `True` si ≥ 1 foco FIRMS con confianza suficiente cayó dentro de la celda ese día |

**Filtros de calidad aplicados:**
- VIIRS: `confidence >= 30`
- MODIS: `confidence in ('nominal', 'high')` (excluye `'low'`)
- Tipo: solo `type == 0` (presunto foco vegetal; excluye volcanes, tierra desnuda, offshore)

**Prevalencia en el dataset 2022–2024:**
- Celdas-día totales: 58,320,352
- Celdas-día con fuego: 42,121 (0.072%)
- Desbalance positivo/negativo: ~1:1,384

---

## 8. Dataset M1 — tabla final

**Archivo:** `data/processed/m1_dataset.parquet`
**Generado por:** `src/features/build_dataset.py`
**Formato:** Apache Parquet, particionado por año (row groups ~1 año × 53K celdas)
**Tamaño actual (2022–2024):** 58,320,352 filas × 28 columnas

### Esquema completo

| Columna | Tipo | Unidad | Fuente | NaN | Descripción |
|---------|------|--------|--------|-----|-------------|
| `cell_id` | str | — | Grilla | 0 | Identificador único de celda (`r{row}_c{col}`) |
| `date` | date | — | — | 0 | Fecha del registro |
| `fire_occurred` | bool | — | FIRMS | 0 | **Target** — `True` si hubo foco activo en la celda ese día |
| `T2M` | float | °C | NASA POWER | 0 | Temperatura media a 2 m |
| `RH2M` | float | % | NASA POWER | 0 | Humedad relativa media a 2 m |
| `WS10M` | float | m/s | NASA POWER | 0 | Velocidad del viento a 10 m |
| `PRECTOTCORR` | float | mm/día | NASA POWER | 0 | Precipitación acumulada 24 h |
| `fwi` | float | adim. | FWI (calculado) | 0 | Fire Weather Index canadiense |
| `ffmc_val` | float | adim. | FWI (calculado) | 0 | Fine Fuel Moisture Code |
| `dmc_val` | float | adim. | FWI (calculado) | 0 | Duff Moisture Code |
| `dc_val` | float | adim. | FWI (calculado) | 0 | Drought Code |
| `isi_val` | float | adim. | FWI (calculado) | 0 | Initial Spread Index |
| `bui_val` | float | adim. | FWI (calculado) | 0 | Build Up Index |
| `prec_acc7d` | float | mm | NASA POWER | 0 | Precipitación acumulada 7 días |
| `prec_acc14d` | float | mm | NASA POWER | 0 | Precipitación acumulada 14 días |
| `ndvi` | float | adim. | MODIS MOD13Q1 | 100% | NDVI diario interpolado *(pendiente)* |
| `ndvi_lag7` | float | adim. | MODIS MOD13Q1 | 100% | NDVI con lag 7 días *(pendiente)* |
| `ndvi_lag14` | float | adim. | MODIS MOD13Q1 | 100% | NDVI con lag 14 días *(pendiente)* |
| `elevation_m` | float | m | SRTM | 0 | Elevación media (SRTM 30m) |
| `slope_deg` | float | ° | SRTM | 0 | Pendiente media |
| `aspect_deg` | float | ° | SRTM | 0 | Aspecto medio |
| `dist_roads_km` | float | km | OSM | 0 | Distancia al camino más cercano |
| `dist_settlements_km` | float | km | OSM | 0 | Distancia al poblado más cercano |
| `is_protected_area` | bool | — | CONAP/WDPA | 0 | Dentro de área protegida |
| `month` | int | 1–12 | Calculado | 0 | Mes del año |
| `day_of_year` | int | 1–366 | Calculado | 0 | Día del año |
| `year` | int | — | Calculado | 0 | Año |
| `split` | str | — | Calculado | 0 | Partición temporal: `'train'` (2018–2022), `'val'` (2023), `'test'` (2024) |

### Features usadas por el modelo M1 actual

El experimento actual usa 20 features (sin NDVI por estar pendiente):

```
T2M, RH2M, WS10M, PRECTOTCORR,
fwi, ffmc_val, dmc_val, dc_val, isi_val, bui_val,
prec_acc7d, prec_acc14d,
elevation_m, slope_deg, aspect_deg,
dist_roads_km, dist_settlements_km, is_protected_area,
month, day_of_year
```

### Feature importance (modelo preliminar 2022–2024)

Orden por LightGBM split importance:

| Rango | Feature | Interpretación |
|-------|---------|----------------|
| 1 | `day_of_year` | Estacionalidad — temporada seca concentra el 90% de los focos |
| 2 | `is_protected_area` | Patrón diferenciado dentro/fuera de reservas |
| 3 | `dist_settlements_km` | Ignición antrópica — quemas agrícolas cerca de poblados |
| 4 | `fwi` | Condición meteorológica global de peligro |
| 5 | `T2M` | Temperatura — deshidratación de combustible fino |
| 6 | `bui_val` | Acumulación de combustible disponible |
| 7 | `elevation_m` | Topografía correlacionada con tipo de vegetación |
| 8 | `dist_roads_km` | Acceso humano al bosque |
| 9 | `WS10M` | Viento — propagación y secado de combustible |
| 10 | `month` | Estacionalidad (redundante con `day_of_year`, menor importancia) |

*Nota: importancias del modelo preliminar (3 años de datos, 1 árbol efectivo). Se actualizarán con el reentrenamiento 2018–2024.*

---

## Notas generales

### CRS del proyecto

| Uso | CRS | EPSG |
|-----|-----|------|
| Análisis métrico (distancias, áreas) | UTM Zona 16N | 32616 |
| I/O y mapas web | WGS84 geográfico | 4326 |

### Convención de fechas

Todas las fechas son en UTC. Los focos FIRMS nocturnos pertenecen a la fecha UTC de adquisición, que puede diferir en ±1 día de la hora local del Petén (UTC-6).

### Política de NaN

- **Clima:** sin NaN — NASA POWER proporciona datos completos para el bbox.
- **FWI:** sin NaN — calculado iterativamente desde el primer día del período.
- **NDVI:** 100% NaN en el dataset actual — pendiente descarga de MOD13Q1. El modelo M1 actual opera sin NDVI.
- **Imputación en inferencia:** NaN se imputa con la mediana de la columna antes de pasar al modelo (ver `src/models/m1_risk/predict.py`).

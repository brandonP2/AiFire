# Plan: Pipeline de Datos Base — M1 Riesgo
**Goal:** Descargar, reproyectar y unir en una grilla 1 km × 1 km todas las fuentes de datos necesarias para entrenar M1 (focos FIRMS, clima POWER, NDVI MODIS, topografía SRTM).
**Date:** 2026-05-08
**Status:** IN PROGRESS

---

## Dependencias externas

| Servicio | URL | Spike necesario | Requiere cuenta |
|----------|-----|-----------------|-----------------|
| NASA FIRMS API | https://firms.modaps.eosdis.nasa.gov | **SÍ** | Sí — MAP_KEY gratuita |
| NASA POWER API | https://power.larc.nasa.gov | **SÍ** | No |
| MODIS MOD13Q1 (NDVI) | https://lpdaac.usgs.gov | **SÍ** | Sí — Earthdata gratuita |
| SRTM 30m | https://dwtkns.com/srtm30m | No (descarga directa por tile) | No |
| OSM Petén (caminos) | https://download.geofabrik.de/central-america/guatemala.html | No (descarga directa) | No |
| WDPA (áreas protegidas) | https://www.protectedplanet.net | No (descarga directa) | Sí — gratuita |

---

## Stream de trabajo

Las tareas siguen este orden estricto:

```
[TASK-01] Tipos y config
       │
       ▼
[TASK-02] Grilla base          ← todo lo demás depende de esta
       │
       ├──────────────┬─────────────────┬─────────────┐
       ▼              ▼                 ▼             ▼
[GATE-03]        [GATE-04]        [GATE-05]      [TASK-06]
Spike FIRMS     Spike POWER      Spike NDVI      SRTM descarga
       │              │                │             │
       ▼              ▼                ▼             ▼
[TASK-07]        [TASK-08]        [TASK-09]      [TASK-10]
Download FIRMS  Download POWER   Download NDVI  Procesar SRTM
       │              │                │             │
       └──────────────┴────────────────┴─────────────┘
                              │
                       [TASK-11] Capas auxiliares (OSM, WDPA)
                              │
                       [TASK-12] Join a grilla → dataset_m1_raw.parquet
                              │
                       [TASK-13] Tests de integración
```

---

## Task List

### Shared — hacer primero

- [ ] **TASK-01:** Definir tipos Pydantic y configuración global en `src/data/schemas.py` y `src/data/config.py`
  - Input: `.env.example`, `CLAUDE.md` (bbox Petén, CRS, semilla)
  - Output: `src/data/config.py` con `Settings(BaseSettings)` cargando `.env`; `src/data/schemas.py` con dataclasses/TypedDicts para `GridCell`, `FirmsRecord`, `ClimateRecord`, `NdviRecord`
  - Agent: shared
  - Notas: `PETEN_BBOX = (-91.45, 15.88, -89.14, 17.82)` en WGS84; CRS trabajo = `EPSG:32616`

- [ ] **TASK-02:** Construir la grilla 1 km × 1 km del Petén en `src/data/build_grid.py`
  - Input: `Settings.peten_bbox`, `Settings.grid_resolution_km`
  - Output: `data/interim/peten_grid.gpkg` — GeoDataFrame con columnas `cell_id (str)`, `geometry (Polygon 1km²)`, `lon_center`, `lat_center`, en EPSG:32616
  - Agent: shared
  - Notas: usar `shapely.geometry.box` + `geopandas.GeoDataFrame`; aplicar máscara de tierra (excluir cuerpos de agua de OSM); CLI Typer con `--output-path`
  - Dependencias: TASK-01

### Spikes — [GATE] no avanzar sin que pasen

- [ ] **TASK-03 [GATE]:** Spike NASA FIRMS — validar endpoint, auth y shape de respuesta en `scripts/spike_firms.py`
  - Input: `FIRMS_MAP_KEY` de `.env`
  - Output: script ejecutable que imprime `✓ FIRMS OK` + muestra 3 filas del CSV para el bbox del Petén en los últimos 7 días; si falla imprime el error HTTP con código y body
  - Agent: backend
  - Cómo correr: `uv run python scripts/spike_firms.py`
  - URL a probar: `https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/VIIRS_SNPP_NRT/{bbox}/1`

- [ ] **TASK-04 [GATE]:** Spike NASA POWER — validar endpoint y parámetros climáticos en `scripts/spike_power.py`
  - Input: ninguna clave (API pública)
  - Output: script que consulta T2M, RH2M, WS10M, PRECTOTCORR para 1 punto del Petén (lat=16.9, lon=-90.2) en un rango de 7 días e imprime el JSON de respuesta
  - Agent: backend
  - Parámetros a validar: `parameters=T2M,RH2M,WS10M,PRECTOTCORR`, `community=RE`, `format=JSON`

- [ ] **TASK-05 [GATE]:** Spike MODIS NDVI (AppEEARS / LPDAAC) — validar descarga de MOD13Q1 en `scripts/spike_ndvi.py`
  - Input: `EARTHDATA_USER`, `EARTHDATA_PASS` de `.env`
  - Output: script que descarga 1 tile HDF (h09v07 cubre Petén) de MOD13Q1 para 2024-01-01 y verifica que el archivo existe y pesa > 1 MB
  - Agent: backend
  - Alternativa si AppEEARS falla: usar Google Earth Engine (`earthengine-api`) como fallback documentado en el spike

### Backend — descarga e ingesta

- [ ] **TASK-07:** Implementar `src/data/download_firms.py` — descarga de focos activos FIRMS
  - Input: TASK-03 pasado; `Settings`; rango de fechas como args CLI
  - Output: `data/raw/firms/firms_{year}.csv` — columnas: `latitude`, `longitude`, `acq_date`, `acq_time`, `confidence`, `bright_ti4`, `satellite`; CLI: `--start YYYY-MM-DD --end YYYY-MM-DD --source VIIRS_SNPP_NRT|MODIS_NRT`
  - Agent: backend
  - Notas: descargar por año para evitar timeouts; reintentar 3× con backoff exponencial; loggear con loguru
  - Dependencias: TASK-01, TASK-03

- [ ] **TASK-08:** Implementar `src/data/download_power.py` — descarga de clima diario NASA POWER
  - Input: TASK-04 pasado; `Settings`; grilla de centroides de `peten_grid.gpkg`
  - Output: `data/raw/power/power_{year}.parquet` — columnas: `date`, `lat`, `lon`, `T2M`, `RH2M`, `WS10M`, `PRECTOTCORR`
  - Agent: backend
  - Notas: NASA POWER acepta punto único o bbox; usar bbox del Petén con resolución nativa de 0.5°; descargar por año; guardar en parquet (más eficiente que CSV para series temporales largas)
  - Dependencias: TASK-01, TASK-04

- [ ] **TASK-09:** Implementar `src/data/download_modis_ndvi.py` — descarga y recorte de MOD13Q1
  - Input: TASK-05 pasado; `Settings`; año como arg CLI
  - Output: `data/raw/ndvi/MOD13Q1_{year}_h09v07.hdf` (crudo) + `data/interim/ndvi/ndvi_{year}.tif` (recortado al Petén, EPSG:32616, 250m)
  - Agent: backend
  - Notas: MOD13Q1 es composición 16 días; interpolar a diario en el paso de features; tile h09v07 cubre todo el Petén; usar `rasterio` + `pyproj` para reproyectar
  - Dependencias: TASK-01, TASK-05

- [ ] **TASK-10:** Implementar `src/data/download_srtm.py` — descarga y mosaico de topografía SRTM
  - Input: `Settings.peten_bbox`; sin credenciales
  - Output: `data/interim/topography/srtm_peten.tif` — elevación en metros, EPSG:32616, 30m resolución, recortado al bbox del Petén
  - Agent: backend
  - Notas: tiles necesarios: N15W090, N15W091, N16W090, N16W091, N17W090, N17W091; URL base: `https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/TIFF/`; usar `rasterio.merge.merge` para mosaico; calcular pendiente y aspecto con `richdem` o numpy derivado
  - Dependencias: TASK-01

- [ ] **TASK-11:** Descargar capas auxiliares estáticas en `src/data/download_static.py`
  - Input: `Settings`
  - Output:
    - `data/raw/osm/guatemala-latest.osm.pbf` (Geofabrik) → filtrar caminos → `data/interim/static/roads_peten.gpkg`
    - `data/raw/wdpa/WDPA_Guatemala.gpkg` → filtrar Petén → `data/interim/static/protected_areas_peten.gpkg`
  - Agent: backend
  - Notas: usar `pyosmium` o `osmium` CLI para filtrar el PBF; las capas estáticas no cambian entre runs — verificar si ya existen antes de descargar
  - Dependencias: TASK-01

### Integración — join a grilla

- [ ] **TASK-12:** Implementar `src/data/build_dataset_m1.py` — unir todas las fuentes a la grilla
  - Input: `peten_grid.gpkg` + todos los outputs de TASK-07 a TASK-11
  - Output: `data/processed/dataset_m1_raw.parquet` — una fila por `(cell_id, date)`, columnas:
    - Target: `fire_occurred (bool)` — 1 si hay foco FIRMS en la celda ese día
    - Features clima: `T2M`, `RH2M`, `WS10M`, `PRECTOTCORR` (interpolados de POWER al centroide de cada celda)
    - Features NDVI: `ndvi`, `ndvi_lag7`, `ndvi_lag14` (interpolación temporal de MOD13Q1 16d → diario)
    - Features topo: `elevation_m`, `slope_deg`, `aspect_deg` (muestreo del raster SRTM en el centroide)
    - Features estáticas: `dist_roads_km`, `dist_settlements_km`, `is_protected_area (bool)`
    - Tiempo: `month`, `day_of_year`, `year`
  - Agent: shared
  - Notas: el join espacial entre FIRMS (puntos) y grilla usa `geopandas.sjoin`; interpolar POWER por IDW o vecino más cercano; guardar en parquet con compresión snappy; loggear estadísticas de desbalance (% de filas con `fire_occurred=True`)
  - Dependencias: TASK-02, TASK-07, TASK-08, TASK-09, TASK-10, TASK-11

### Tests

- [ ] **TASK-13:** Escribir tests de integración del pipeline en `tests/data/`
  - Input: fixtures pequeñas (subset de 10×10 celdas, 30 días de datos sintéticos)
  - Output:
    - `tests/data/test_build_grid.py` — verifica CRS, número de celdas, ausencia de geometrías nulas
    - `tests/data/test_download_firms.py` — mock de la API HTTP; verifica columnas y tipos del CSV resultante
    - `tests/data/test_build_dataset_m1.py` — verifica que el join produce el schema correcto y que `fire_occurred` tiene valores 0/1
  - Agent: shared
  - Notas: usar `pytest-mock` para parchear `httpx.get`; usar `tmp_path` de pytest para paths temporales; marcar tests de descarga real con `@pytest.mark.network`
  - Dependencias: TASK-02, TASK-07, TASK-12

---

## Checklist de validación del plan

- [x] Ninguna tarea tarda más de 5 minutos para un agente enfocado
- [x] Todo servicio externo tiene un spike `[GATE]` antes de la integración
- [x] Las tareas shared (TASK-01, TASK-02) van antes que todas las demás
- [x] Cada tarea tiene output verificable y concreto
- [x] Los tests cubren el schema del dataset final y los mocks de APIs

---

## Cómo ejecutar

```bash
# 1. Setup (una vez)
uv sync --extra dev
cp .env.example .env   # completar FIRMS_MAP_KEY, EARTHDATA_USER, EARTHDATA_PASS

# 2. Spikes — deben pasar ANTES de seguir
uv run python scripts/spike_firms.py
uv run python scripts/spike_power.py
uv run python scripts/spike_ndvi.py

# 3. Pipeline completo
uv run python -m src.data.build_grid
uv run python -m src.data.download_firms --start 2018-01-01 --end 2024-12-31
uv run python -m src.data.download_power --start 2018-01-01 --end 2024-12-31
uv run python -m src.data.download_modis_ndvi --year 2018  # repetir por año
uv run python -m src.data.download_srtm
uv run python -m src.data.download_static
uv run python -m src.data.build_dataset_m1

# 4. Tests
uv run pytest tests/data/ -v
uv run pytest tests/data/ -m "not network" -v  # sin llamadas reales a APIs
```

---

## Próximo paso

Una vez que `dataset_m1_raw.parquet` exista y los tests pasen:
→ Pasar a `docs/plans/02-features-m1.md` (feature engineering: FWI, lags, distancias) y luego `docs/plans/03-model-m1.md` (entrenamiento XGBoost baseline).

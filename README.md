# PetenFire

**Sistema Integrado de Predicción y Detección de Incendios Forestales para el Petén, Guatemala**

PetenFire es una plataforma de ciencia de datos que combina datos satelitales abiertos, climatología y aprendizaje automático para reducir el impacto de los incendios forestales en el departamento del Petén — hogar de la Reserva de Biosfera Maya, el Parque Nacional Tikal y la Sierra del Lacandón.

---

## Visión

Pasar de una respuesta **reactiva** a una **predictiva** frente al fuego, dando a CONRED, INAB, CONAP y MARN herramientas operativas para anticipar, simular y detectar incendios en tiempo casi real.

## Componentes

El sistema se compone de **tres modelos integrados**:

| Modelo | Tarea | Salida | Enfoque base |
|--------|-------|--------|--------------|
| **M1 — Riesgo** | Predecir probabilidad de ignición por celda y día | Mapa de riesgo (1, 3, 7 días) | XGBoost / LightGBM |
| **M2 — Propagación** | Simular avance de un incendio activo | Raster de probabilidad de quema (6h, 12h, 24h) | Autómata celular tipo FARSITE / ML supervisado |
| **M3 — Detección** | Confirmar focos activos en imágenes satelitales | Clasificación + bounding box | CNN ligera (ResNet18) / YOLOv8 |

## Cobertura

- **Geográfica:** Departamento del Petén (~35,854 km²)
- **Resolución espacial:** cuadrícula de 1 km × 1 km
- **Resolución temporal:** diaria, horizonte de 1, 3 y 7 días
- **Datos de entrenamiento:** 2018–2024

## Fuentes de datos

NASA FIRMS · MODIS MCD64A1 · MOD13Q1 (NDVI) · NASA POWER · Copernicus ERA5 · Sentinel-2 · Landsat-8/9 · SRTM · ESA WorldCover · OpenStreetMap · CONAP / WDPA.

## Estructura del proyecto

```
petenfire/
├── data/              # raw / interim / processed (no versionado)
├── notebooks/         # EDA y prototipos
├── src/
│   ├── data/          # Descarga e ingesta
│   ├── features/      # Feature engineering
│   ├── models/        # M1 riesgo, M2 propagación, M3 detección
│   ├── evaluation/    # Métricas y validación temporal
│   └── viz/           # Mapas y visualizaciones
├── dashboard/         # Aplicación Streamlit
├── tests/             # Pytest
├── docs/              # PRD, diccionario de datos, model cards
└── reports/           # Salidas para stakeholders
```

## Inicio rápido

Requisitos: **Python ≥ 3.11** y [`uv`](https://docs.astral.sh/uv/).

```bash
# Clonar e instalar
git clone <repo-url> petenfire
cd petenfire
uv sync --extra dev

# Variables de entorno
cp .env.example .env   # completar credenciales

# Verificar instalación
uv run ruff check .
uv run pytest -m "not slow and not network"
```

### Variables de entorno requeridas

| Variable | Fuente | Necesaria para |
|----------|--------|----------------|
| `FIRMS_MAP_KEY` | [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/api/area/) | Descarga de focos activos |
| `EARTHDATA_USER` / `EARTHDATA_PASS` | [NASA Earthdata](https://urs.earthdata.nasa.gov/) | MODIS, SRTM, burned area |
| `CDS_API_KEY` | [Copernicus CDS](https://cds.climate.copernicus.eu/) | ERA5 (clima alta resolución) |
| `GEE_SERVICE_ACCOUNT` / `GEE_PRIVATE_KEY_PATH` | [Google Earth Engine](https://earthengine.google.com/) | NDVI Sentinel-2 (M3) |

NASA POWER (clima diario) y OpenStreetMap no requieren credenciales.

---

## Pipeline de datos — M1

El pipeline completo para reproducir el dataset y el modelo M1 desde cero:

```bash
# 1. Construir la grilla 1 km × 1 km del Petén (EPSG:32616)
uv run python -m src.data.build_grid

# 2. Descargar features estáticas (SRTM, OSM caminos/poblados, áreas protegidas)
uv run python -m src.data.download_static

# 3. Descargar clima diario NASA POWER (2018-2024, ~2 min)
uv run python -m src.data.download_power --start 2018-01-01 --end 2024-12-31

# 4. Descargar focos activos NASA FIRMS (2018-2024, ~90 min en paralelo)
uv run python -m src.data.download_firms --start 2018-01-01 --end 2024-12-31 --source VIIRS_SNPP_SP
uv run python -m src.data.download_firms --start 2018-01-01 --end 2024-12-31 --source MODIS_SP

# 5. Calcular features climáticas + FWI (Canadian Fire Weather Index)
uv run python -m src.features.climate_features --start 2018-01-01 --end 2024-12-31

# 6. Generar etiquetas binarias de fuego desde FIRMS
uv run python -m src.features.fire_labels --start 2018-01-01 --end 2024-12-31

# 7. Construir dataset final (une features + etiquetas + splits temporales)
uv run python -m src.features.build_dataset

# 8. Entrenar M1 LightGBM con validación temporal estricta
uv run python -m src.models.m1_risk.train --model lightgbm --n-estimators 500

# 9. Generar mapa de riesgo para una fecha
uv run python -m src.models.m1_risk.predict --date 2024-03-15
```

**Splits temporales:** train 2018–2022 · val 2023 · test 2024. Nunca K-fold aleatorio.

---

## Dashboard

```bash
uv run streamlit run dashboard/app.py
# → http://localhost:8501
```

Tres pestañas:
- **Mapa de Riesgo** — heatmap interactivo del Petén por fecha, con incendios FIRMS superpuestos
- **Métricas del Modelo** — AUC-ROC, AUC-PR, feature importance (LightGBM split importance)
- **Análisis FIRMS** — serie temporal mensual, estacionalidad, distribución FRP por fuente

---

## Desarrollo

```bash
# Lint + format
uv run ruff check . --fix
uv run ruff format .

# Type checking
uv run mypy src/

# Tests con cobertura
uv run pytest

# Solo tests rápidos (sin red ni datos grandes)
uv run pytest -m "not slow and not network"
```

---

## Estado actual (M1)

| Componente | Estado |
|------------|--------|
| Grilla 1 km² Petén | Completo — 53,212 celdas |
| Features estáticas (elevación, pendiente, distancias, áreas protegidas) | Completo |
| Clima diario NASA POWER 2022–2024 | Completo |
| FWI canadiense (FFMC, DMC, DC, ISI, BUI, FWI) | Completo |
| Focos activos FIRMS 2022–2024 (MODIS + VIIRS) | Completo |
| Dataset M1 2022–2024 | Completo — 58M filas |
| M1 LightGBM (entrenado con 2022–2024) | Completo — AUC-ROC 0.85 / F1 0.00 (modelo preliminar) |
| Clima + FIRMS 2018–2021 (ampliar entrenamiento) | Descargando |
| M1 reentrenado 2018–2024 (objetivo) | Pendiente |
| M2 Propagación | Pendiente |
| M3 Detección CNN | Pendiente |

El F1=0.0 del modelo preliminar refleja entrenamiento con solo 3 años y 1 árbol efectivo. El reentrenamiento con 7 años de datos (2018–2024) es el paso inmediato.

---

## Métricas objetivo (MVP)

| Componente | Métrica | Meta |
|------------|---------|------|
| M1 | AUC-ROC / Recall en zonas de alto riesgo | ≥ 0.80 / ≥ 0.75 |
| M2 | Error de área quemada predicha vs. real | < 30% |
| M3 | Precisión / Recall en focos activos | ≥ 0.85 / ≥ 0.80 |
| Producto | Tiempo de generación del mapa diario | < 15 min |

## Cronograma (12 semanas)

1. **Sem 1** — Setup, PRD, descarga inicial
2. **Sem 2–3** — EDA y dataset M1
3. **Sem 4–5** — M1 baseline + iteración
4. **Sem 6–7** — M2 propagación
5. **Sem 8–9** — M3 detección (CNN)
6. **Sem 10–11** — Dashboard integrado
7. **Sem 12** — Documentación, model cards, presentación

## Stakeholders

- **Primarios:** CONRED, INAB
- **Secundarios:** MARN, CONAP, municipalidades del Petén
- **Terciarios:** WCS, Defensores de la Naturaleza, comunidad investigadora

## Decision Log

| Fecha | Decisión | Razón |
|-------|----------|-------|
| 2026-05 | **LightGBM sobre XGBoost como modelo principal M1** | Más rápido en tabular con alta cardinalidad (53K celdas × 365 días). Ambos están implementados; LightGBM elegido por defecto tras pruebas iniciales. |
| 2026-05 | **Validación temporal estricta (no K-fold)** | Series temporales con autocorrelación temporal y espacial: K-fold aleatorio daría AUC inflado ~0.10 por leakage futuro→pasado. Split fijo 2018-2022 / 2023 / 2024. |
| 2026-05 | **Calibración isotónica post-hoc** | LightGBM con SMOTE sobreestima probabilidades en la clase positiva. La calibración isotónica corrige el rango [0,1] para que `risk_prob` sea interpretable como probabilidad real. |
| 2026-05 | **SMOTE conservador (strategy=0.1) en lugar de oversampling agresivo** | Desbalance ~0.07% (42K focos vs. 58M celdas-día). SMOTE a 10% genera suficientes ejemplos positivos sin distorsionar la distribución de features climáticos continuos. |
| 2026-05 | **CalibratedModel extraído a `src/models/m1_risk/__init__.py`** | Cuando train.py se ejecuta como `__main__`, joblib serializa la clase como `__main__._CalibratedModel`, rompiendo la deserialización en otros contextos (dashboard, scripts). Mover la clase a un módulo estable resuelve el problema sin patches. |
| 2026-05 | **Pushdown filter de PyArrow en el dashboard** | El dataset tiene 58M filas. Leer el parquet completo para filtrar por fecha tardaba >2 min. `pd.read_parquet(filters=[('date','=', date)])` baja la lectura a ~0.6s usando el índice de row groups de Parquet. |
| 2026-05 | **Heatmap normalizado por percentil 99 en el dashboard** | El modelo preliminar predice probabilidades en rango [0, 0.001]. Folium HeatMap con valores absolutos tan bajos no muestra nada. Normalizar por `p99` hace el mapa relativo legible mientras el modelo maduro no esté disponible. |
| 2026-05 | **NASA POWER en lugar de ERA5 para clima diario** | NASA POWER cubre el período 1981–presente con API REST pública sin registro (vs. CDS API con cuota). Resolución ~50 km suficiente para features climáticas agregadas a nivel de celda 1 km. ERA5 queda como fuente adicional para features de alta resolución si se necesitan. |

---

## Licencia

MIT — ver `LICENSE`.

## Documentación adicional

- [`docs/prd.md`](docs/prd.md) — Product Requirements Document completo
- [`docs/data_dictionary.md`](docs/data_dictionary.md) — Diccionario de datos
- [`docs/model_cards/`](docs/model_cards/) — Tarjetas de modelo (M1, M2, M3)

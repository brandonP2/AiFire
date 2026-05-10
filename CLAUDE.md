# CLAUDE.md — PetenFire

Instrucciones operativas para Claude Code al trabajar en este repositorio. Estas reglas tienen prioridad sobre los defaults globales.

---

## 1. Contexto del dominio

**PetenFire** es un sistema de ciencia de datos para **predecir, simular y detectar incendios forestales** en el departamento del **Petén, Guatemala** (~35,854 km², hogar de la Reserva de Biosfera Maya, Parque Nacional Tikal y Sierra del Lacandón).

- **Temporada crítica:** marzo–mayo (estación seca).
- **Stakeholders:** CONRED (operativo), INAB (forestal), MARN, CONAP (áreas protegidas), municipalidades del Petén.
- **Tres modelos integrados:**
  - **M1 — Riesgo:** clasificación de probabilidad de ignición por celda-día (XGBoost / LightGBM).
  - **M2 — Propagación:** simulación de avance del fuego (autómata celular tipo FARSITE o ML supervisado).
  - **M3 — Detección:** visión por computadora sobre Sentinel-2 / Landsat (ResNet18 / YOLOv8).
- **Resolución de trabajo:** cuadrícula **1 km × 1 km**, paso **diario**, horizontes de 1, 3 y 7 días.
- **Datos de entrenamiento:** 2018–2024.

### Fuentes de datos (todas abiertas)

| Dato | Fuente | Notas |
|------|--------|-------|
| Focos activos | **NASA FIRMS** (MODIS + VIIRS) | API key gratuita |
| Burned area | **MODIS MCD64A1** | NASA Earthdata |
| Clima diario | **NASA POWER** | API pública |
| Clima alta resolución | **Copernicus ERA5** | CDS API (cdsapi) |
| NDVI | **MODIS MOD13Q1** | Earthdata |
| Topografía | **SRTM 30m** | USGS |
| Uso de suelo | **ESA WorldCover** / MapBiomas | Descarga directa |
| Imágenes ópticas | **Sentinel-2, Landsat-8/9** | Copernicus, USGS, GEE, STAC |
| Caminos / poblados | **OpenStreetMap** | Geofabrik |
| Áreas protegidas | **CONAP / WDPA** | Descarga directa |

### Vocabulario del dominio (úsalo en código y docs)

- **Foco activo (active fire):** detección térmica de FIRMS, no necesariamente un incendio confirmado.
- **Burned area:** polígono o píxel con quema confirmada a posteriori (MCD64A1).
- **Celda:** unidad espacial de 1 km × 1 km en la grilla del Petén.
- **Horizonte:** ventana temporal hacia adelante de la predicción (1d, 3d, 7d).
- **NDVI / NBR:** índices de vegetación / quema usados como features.
- **FWI:** Fire Weather Index (Canadian system) — feature compuesta de clima.

---

## 2. Stack tecnológico

| Capa | Tecnología | Versión mínima |
|------|------------|----------------|
| Lenguaje | **Python** | 3.11 |
| Gestor de entorno / paquetes | **uv** | última estable |
| Geoespacial vectorial | **geopandas**, shapely, fiona, pyproj | 1.0+ |
| Geoespacial raster | **rasterio**, **xarray**, rioxarray | — |
| ML clásico | **xgboost**, **lightgbm**, scikit-learn, imbalanced-learn | — |
| Deep Learning (M3) | **pytorch**, torchvision, ultralytics (YOLOv8) | torch ≥ 2.2 |
| APIs satelitales | cdsapi, earthengine-api, pystac-client, planetary-computer | — |
| Dashboard | **streamlit**, folium, plotly | — |
| Calidad | **ruff** (lint + format), **pytest** (+ cov), mypy | — |
| Logging / CLI | loguru, typer, pydantic-settings | — |

**No introduzcas dependencias nuevas sin justificación.** Si una librería ya cubre el caso, úsala.

---

## 3. Convenciones de código

### Obligatorias

- **Type hints en TODA función pública** (parámetros y retorno). Usa `from __future__ import annotations` y sintaxis moderna (`list[str]`, `dict[str, int]`, `X | None`).
- **Docstrings estilo Google** en módulos, clases y funciones públicas:
  ```python
  def compute_fwi(temp: float, rh: float, wind: float, rain: float) -> float:
      """Calcula el Fire Weather Index canadiense.

      Args:
          temp: Temperatura en °C a las 12:00 local.
          rh: Humedad relativa (%).
          wind: Velocidad del viento (km/h) a 10 m.
          rain: Precipitación acumulada 24 h (mm).

      Returns:
          Índice FWI adimensional. Valores >30 indican peligro alto.

      Raises:
          ValueError: Si `rh` no está en [0, 100].
      """
  ```
- **Formato:** `ruff format` (line-length 100). No mezclar comillas — usar `"`.
- **Imports:** ordenados por `ruff` (regla `I`). Nunca imports relativos profundos (`from ...x import y`).
- **Naming:** `snake_case` para funciones/variables, `PascalCase` para clases, `UPPER_SNAKE` para constantes.
- **Configuración:** todo path/credencial vía `pydantic-settings` leyendo `.env`. **Nunca hardcodear** rutas absolutas ni claves.
- **Logging:** usar `loguru`, **nunca `print`** en código de `src/`.
- **Errores:** levantar excepciones específicas (`ValueError`, `FileNotFoundError`, custom). **No silenciar** excepciones con `except: pass`.

### Geoespacial — reglas críticas

- **Siempre declarar y validar el CRS** al cargar/guardar rasters y vectoriales.
- **CRS de trabajo del proyecto:** `EPSG:32616` (UTM 16N) para análisis métrico; `EPSG:4326` solo para I/O y mapas web.
- **No usar `geopandas.read_file` sin chequear el CRS** del archivo.
- Rasters grandes: usar **lectura por ventanas** (`rasterio.windows`) o `rioxarray` con `chunks=`.
- Cierra siempre los datasets (`with rasterio.open(...) as src:`).

### Notebooks

- Solo para **EDA y prototipos**. Lógica reutilizable → mover a `src/`.
- Antes de commitear: `nbstripout` para limpiar outputs.
- Numerar: `01_eda_firms.ipynb`, `02_eda_climate.ipynb`, etc.

---

## 4. Comandos esenciales

Todos los comandos asumen el venv gestionado por `uv` (no hace falta activarlo).

```bash
# Setup inicial
uv sync --extra dev

# Tests
uv run pytest                           # suite completa con cobertura
uv run pytest -m "not slow and not network"   # solo rápidos / offline
uv run pytest tests/test_m1_risk.py -v        # un archivo
uv run pytest -k "fwi" -v                     # por keyword

# Lint y formato
uv run ruff check .                     # lint
uv run ruff check . --fix               # lint + autofix
uv run ruff format .                    # formato

# Type checking
uv run mypy src/

# Dashboard
uv run streamlit run dashboard/app.py

# Notebooks
uv run jupyter lab

# Pipelines (CLI Typer)
uv run python -m src.data.download_firms --start 2024-01-01 --end 2024-12-31
uv run python -m src.data.build_grid
```

**Si un comando falla 2 veces seguidas: PARA, diagnostica, reporta.** No hagas brute-force.

---

## 5. Estructura de carpetas

```
petenfire/
├── CLAUDE.md                  # Este archivo — instrucciones para Claude Code
├── README.md                  # Documentación pública del proyecto
├── pyproject.toml             # Dependencias + config de ruff/pytest/mypy
├── .env.example               # Plantilla de variables de entorno
├── .gitignore
│
├── data/                      # Datasets — NO versionado (ver reglas §6)
│   ├── raw/                   # Datos crudos descargados, inmutables
│   ├── interim/               # Procesados intermedios (reproyectados, recortados)
│   └── processed/             # Datasets finales listos para entrenar
│
├── notebooks/                 # EDA y prototipos numerados (01_, 02_, …)
│
├── src/                       # Código fuente reutilizable — paquete instalable
│   ├── data/                  # Descarga, ingesta y construcción de la grilla
│   │                          # (download_firms.py, download_power.py, build_grid.py, …)
│   ├── features/              # Feature engineering (FWI, NDVI lags, distancias, …)
│   ├── models/
│   │   ├── m1_risk/           # Predicción de riesgo: training, inferencia, calibración
│   │   ├── m2_spread/         # Propagación: simulador + ML supervisado
│   │   └── m3_detection/      # CNN/YOLO para focos activos en Sentinel-2/Landsat
│   ├── evaluation/            # Métricas, validación temporal, splits espaciales
│   └── viz/                   # Generación de mapas, figuras y tiles
│
├── dashboard/                 # Aplicación Streamlit (entrypoint: app.py)
│
├── tests/                     # Pytest — espejo de la estructura de src/
│
├── docs/                      # Documentación técnica
│   ├── prd.md                 # Product Requirements Document
│   ├── data_dictionary.md     # Definición de cada feature y fuente
│   └── model_cards/           # Una tarjeta por modelo (M1, M2, M3)
│
└── reports/                   # Salidas para stakeholders (figuras, mapas HTML, PDFs)
```

**Reglas de ubicación:**
- Lógica reutilizable → `src/`, **nunca** en notebooks.
- Tests reflejan la estructura: `src/features/fwi.py` → `tests/features/test_fwi.py`.
- Artefactos de entrenamiento (modelos `.pkl`, `.pt`) → fuera de git, en `models_artifacts/` o tracking server.

---

## 6. Reglas inviolables

### Datos
- **NUNCA commitear `data/raw/`, `data/interim/`, ni `data/processed/`.** El `.gitignore` ya los bloquea — no lo modifiques para incluirlos.
- **NUNCA commitear:** `.env`, `.cdsapirc`, `.netrc`, `.earthdatarc`, claves de servicio (`*.json` de GEE), tokens de FIRMS.
- Si necesitas compartir un dataset, sube a almacenamiento externo (S3, GCS, Drive institucional) y documenta el link en `docs/data_dictionary.md`.
- Datos crudos son **inmutables**: si necesitas transformarlos, escribe a `interim/` o `processed/`.

### Calidad y merge
- **Validar siempre con tests antes de mergear.** Mínimo: `uv run pytest` + `uv run ruff check .` en verde.
- Cobertura objetivo: **≥ 70%** sobre `src/` (excluye `__init__.py`).
- Cualquier feature nueva en `src/features/` o `src/models/` requiere **al menos un test unitario**.
- Cambios en pipelines de datos requieren **un test de integración** (puede usar fixtures pequeñas en `tests/fixtures/`).
- No mergear con tests `xfail` o `skip` no justificados por comentario.

### Modelos y validación
- **Validación temporal estricta** para M1: train 2018–2022, val 2023, test 2024. **Nunca** hacer K-fold aleatorio sobre series temporales.
- Para M3: split por **escena/tile**, no por píxel, para evitar leakage espacial.
- Cada modelo entrenado debe tener una **model card** en `docs/model_cards/` antes de considerarse "listo".
- Reportar siempre métricas con **intervalos de confianza** o desviación estándar sobre múltiples runs.

### Reproducibilidad
- **Semilla fija** (`RANDOM_SEED` desde `.env`, default 42) en todo entrenamiento.
- Versionar configs de experimento (YAML) en el repo, no los outputs.
- Loggear versión de paquetes clave (xgboost, torch) en cada run.

### Seguridad
- Sanitizar inputs en cualquier endpoint del dashboard (Streamlit forms).
- Errores al usuario en el dashboard: mensajes amigables, **nunca** stack traces.
- Validar bbox y fechas en CLIs con `pydantic` antes de hacer requests a APIs externas.

### Acciones destructivas
- **Nunca** ejecutar `rm -rf data/`, `git reset --hard`, `git push --force`, ni borrar branches sin confirmación explícita del usuario.
- Antes de modificar config (`pyproject.toml`, `.gitignore`): explicar el cambio propuesto.

---

## 7. Workflow recomendado por tarea

| Tarea | Pasos |
|-------|-------|
| Nueva fuente de datos | 1) Spike en notebook → 2) Mover a `src/data/download_X.py` con CLI Typer → 3) Test con fixture pequeño → 4) Documentar en `data_dictionary.md` |
| Nuevo feature | 1) Implementar en `src/features/` con type hints + docstring → 2) Test unitario con caso conocido → 3) Añadir a pipeline de dataset |
| Iteración de modelo | 1) Config YAML del experimento → 2) Train script en `src/models/mX_*/` → 3) Evaluar con métricas estándar → 4) Actualizar model card |
| Cambio en dashboard | 1) Probar localmente con `uv run streamlit run dashboard/app.py` → 2) Verificar con datos sintéticos si los reales no están disponibles |

---

## 8. Cuándo preguntar al usuario

- Antes de **introducir una dependencia nueva** que no esté en el stack listado.
- Antes de **descargar datasets >1 GB** (puede tardar y consumir cuota de APIs).
- Antes de **modificar la grilla, CRS de trabajo o resolución** (afecta todos los modelos).
- Si los **resultados de validación están muy por debajo** de las metas (AUC <0.70, etc.) — no "tunear hasta que pase".
- Cuando un dato esperado **no exista o esté corrupto** — no inventar valores.

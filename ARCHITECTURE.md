# ARCHITECTURE.md — PetenFire

Decisiones de arquitectura tomadas en Fase 0. Cada sección explica qué se eligió, por qué, y qué alternativas se descartaron con justificación específica.

---

## 1. Stack elegido

| Capa | Tecnología | Justificación |
|------|------------|---------------|
| Lenguaje | **Python 3.11** | Ecosistema dominante en geoespacial y ML; librerías sin alternativa equivalente en otros lenguajes (rasterio, geopandas, xarray) |
| Gestión de entorno | **uv** | 10–100× más rápido que pip/conda; lockfile reproducible; soporte nativo de `pyproject.toml` |
| Geoespacial vectorial | **geopandas + shapely 2.0** | Estándar de facto; shapely 2.0 usa GEOS con vectorización SIMD |
| Geoespacial raster | **rasterio + xarray + rioxarray** | rasterio para I/O de bajo nivel; xarray para datasets N-dimensionales con coordenadas; rioxarray como puente |
| ML clásico | **XGBoost + LightGBM** | Ambos soportan `scale_pos_weight` para desbalance; LightGBM más rápido en datasets grandes; XGBoost más documentado para calibración |
| Imbalanced learning | **imbalanced-learn** | SMOTE y variantes para M1; alternativa a focal loss sin requerir PyTorch |
| Deep Learning | **PyTorch + torchvision + Ultralytics YOLOv8** | PyTorch para fine-tuning de ResNet18; YOLOv8 si se requieren bounding boxes |
| APIs satelitales | **earthengine-api + pystac-client + cdsapi** | GEE para chips Sentinel-2 a escala; STAC/Planetary Computer como alternativa sin cuenta Google; cdsapi para ERA5 |
| Dashboard | **Streamlit + folium + plotly** | Streamlit: prototipado rápido, sin frontend JS; folium: mapas interactivos nativos en Python; plotly: gráficos interactivos |
| Calidad | **ruff + pytest + mypy** | ruff reemplaza flake8 + isort + black en un binario 100× más rápido; pytest con fixtures geoespaciales pequeñas |
| Logging | **loguru** | API más simple que `logging` estándar; compatible con estructurado JSON si se necesita en producción |
| CLI | **typer + pydantic-settings** | Typer genera CLI desde type hints; pydantic-settings valida `.env` con tipos en tiempo de importación |
| Tracking de experimentos | **MLflow** (recomendado, no obligatorio en MVP) | Logging de parámetros, métricas y artefactos por run; UI local; sin costo |

---

## 2. Alternativas evaluadas y descartadas

### 2.1 Framework de orquestación del pipeline

**Opción A — Prefect (DESCARTADA)**
- Qué ofrecía: UI de monitoreo, reintentos automáticos, scheduling declarativo.
- Por qué descartada: sobrecarga de infraestructura (servidor de agentes) innecesaria para un pipeline que corre una vez al día en un equipo de 1 persona. Añade ~200 MB de dependencias.

**Opción B — Apache Airflow (DESCARTADA)**
- Por qué descartada: requiere base de datos, scheduler y worker separados. Diseñado para equipos con docenas de DAGs. Overkill total para este proyecto.

**Elegida — Makefile + scripts CLI con Typer**
- Cada paso del pipeline es un módulo Python con CLI Typer (`uv run python -m src.data.download_firms`).
- Un `Makefile` encadena los pasos con dependencias explícitas.
- Sin infraestructura adicional. Reproducible en cualquier máquina con `uv`.
- Trade-off aceptado: sin reintentos automáticos ni monitoreo visual. Mitigación: logging con loguru a archivo.

---

### 2.2 Fuente de imágenes satelitales para M3

**Opción A — Descarga directa de archivos `.SAFE` de Copernicus (DESCARTADA)**
- Sentinel-2 L2A para Petén en temporada seca: ~4–8 escenas × ~800 MB = varios GB por fecha.
- Para 7 años de temporadas seca (marzo–mayo): ~630 GB mínimo. Inmanejable localmente.
- Por qué descartada: costo de almacenamiento y tiempo de descarga prohibitivos.

**Opción B — Google Earth Engine (ELEGIDA PRIMARIA)**
- Permite generar chips 64×64 sobre zonas de interés sin descargar el `.SAFE` completo.
- Cruzar con FIRMS para etiquetado automático directamente en GEE.
- Exportar solo los chips necesarios a Google Drive (~GB, no TB).
- Trade-off: requiere cuenta de GEE (gratuita para investigación) y autenticación OAuth.

**Opción C — STAC + Planetary Computer (ELEGIDA ALTERNATIVA)**
- Si GEE tiene fricción de auth o cuotas, `pystac-client` + `stackstac` + Planetary Computer da acceso equivalente sin cuenta Google.
- Suscription key gratuita de Microsoft.
- Ambas opciones están en `pyproject.toml`.

---

### 2.3 Modelo base para M1

**Opción A — Random Forest (DESCARTADA)**
- Más lento en inferencia para grillas grandes; menor rendimiento que boosting en datos tabulares con features mixtas (continuas + categóricas + espaciales).
- Descartada en favor de XGBoost/LightGBM que dominan benchmarks en datos tabulares.

**Opción B — Red neuronal tabular (TabNet, MLP) (DESCARTADA para MVP)**
- Mayor complejidad de entrenamiento y calibración.
- No supera consistentemente a boosting en datasets de tamaño medio con desbalance extremo.
- Reservada para v2 si XGBoost/LightGBM no alcanzan las métricas objetivo.

**Elegida — XGBoost + LightGBM (ensemble o selección por validación)**
- XGBoost: mejor soporte para calibración de probabilidades (Platt scaling).
- LightGBM: 3–10× más rápido en entrenamiento; mejor manejo de features categóricas nativas.
- Estrategia: entrenar ambos, seleccionar por AUC en validación 2023, calibrar el ganador.

---

### 2.4 Modelo base para M2 (Propagación)

**Opción A — FARSITE completo (DESCARTADA)**
- Software de simulación de incendios del USDA. Requiere datos de combustible (LANDFIRE) no disponibles para Guatemala con la resolución necesaria.
- Configuración compleja. No es un modelo Python nativo.

**Opción B — Autómata celular simplificado (ELEGIDA PRIMARIA)**
- Implementado en Python/NumPy. Reglas físicas simples: cada celda se propaga a vecinos con probabilidad función de viento, pendiente y humedad del combustible.
- Interpretable para stakeholders (CONRED puede entender "el fuego se mueve más rápido cuesta arriba y a favor del viento").
- Trade-off: no captura dinámica de combustible en tiempo real.

**Opción C — ML supervisado sobre polígonos MCD64A1 (ELEGIDA COMPLEMENTARIA)**
- Para eventos históricos con polígono de burned area conocido: entrenar un modelo que prediga qué celdas quemaron dado el punto de ignición + condiciones.
- Más datos-driven pero requiere suficientes eventos históricos grandes en el Petén.

---

### 2.5 Dashboard

**Opción A — FastAPI + React (DESCARTADA)**
- Mayor flexibilidad de UI, pero requiere mantener dos bases de código (backend API + frontend JS).
- Tiempo de desarrollo 3–5× mayor para el mismo resultado en MVP.
- Descartada: el equipo es 1 persona con foco en ML, no frontend.

**Opción B — Dash (Plotly) (DESCARTADA)**
- Más customizable que Streamlit para mapas geoespaciales, pero API más verbosa.
- Comunidad más pequeña; menos integraciones nativas con folium.

**Elegida — Streamlit + folium + plotly**
- Componentes en Python puro. Dashboard funcional en horas, no días.
- `streamlit-folium` integra mapas Leaflet directamente.
- Trade-off aceptado: menos control sobre layout y rendimiento con datos muy grandes. Mitigación: cachear resultados con `@st.cache_data`.

---

## 3. Decisiones de datos

### CRS de trabajo
- **Análisis:** `EPSG:32616` (UTM zona 16N) — unidades métricas, distorsión mínima para el Petén.
- **I/O y mapas web:** `EPSG:4326` (WGS84 geográfico).
- **Nunca** mezclar CRS sin reproyección explícita.

### Grilla base
- 1 km × 1 km en UTM 16N.
- Petén bbox (WGS84): `lon [-91.45, -89.14]`, `lat [15.88, 17.82]`.
- Número aproximado de celdas: ~700 × 220 ≈ 154,000 celdas.
- Máscara de tierra: excluir cuerpos de agua (Lago Petén Itzá, Laguna del Tigre, etc.) para reducir ruido.

### Período de entrenamiento
- **Train:** 2018–2022 (5 años)
- **Validación:** 2023 (1 año) — tuning de hiperparámetros y selección de modelo
- **Test:** 2024 (1 año) — evaluación final, **no tocar hasta que el modelo esté congelado**

### Granularidad temporal
- Predicción diaria. Features: valor del día anterior + lags de 3, 7 y 14 días para variables clave.
- Horizonte de predicción: D+1, D+3, D+7. Cada horizonte es un modelo independiente o un output multi-target.

---

## 4. Flujo de datos end-to-end

```
APIs externas                 Pipeline local               Modelos              Dashboard
────────────────              ─────────────────            ────────             ─────────
NASA FIRMS           ──┐
NASA POWER           ──┤──► src/data/download_*.py ──► data/raw/
MODIS MOD13Q1        ──┤
ERA5 (Copernicus)    ──┤
SRTM (USGS)          ──┘

                           ──► src/data/build_grid.py  ──► data/interim/grid.gpkg

data/raw/ + grid            ──► src/features/*.py       ──► data/processed/
                                 (FWI, NDVI lags,              dataset_m1.parquet
                                  distancias, topografía)      dataset_m2.parquet

GEE / STAC                  ──► src/data/download_chips.py ──► data/processed/chips/
(Sentinel-2 tiles)

data/processed/             ──► src/models/m1_risk/train.py  ──► models_artifacts/m1/
                            ──► src/models/m2_spread/sim.py  ──► models_artifacts/m2/
                            ──► src/models/m3_detection/     ──► models_artifacts/m3/
                                 train_cnn.py

models_artifacts/           ──► dashboard/app.py (Streamlit) ──► Usuario (CONRED, INAB…)
```

---

## 5. Validación temporal (anti-leakage)

```
─────────────────────────────────────────────────────────── tiempo ──►
│       TRAIN (2018–2022)        │  VAL (2023)  │  TEST (2024)  │
│  features_t → label_t+horizon  │  tune HP     │  eval final   │
└────────────────────────────────┴──────────────┴───────────────┘
```

- **Nunca** usar K-fold aleatorio sobre la dimensión temporal.
- Features calculadas siempre con datos disponibles en `t` (no `t+1`).
- Para M3: split por escena/tile, no por píxel — evitar leakage espacial.

---

## 6. Seguridad y privacidad

- Sin datos personales en ninguna capa del sistema.
- Credenciales de APIs en `.env` (nunca commiteadas).
- El dashboard en MVP no tiene autenticación — desplegar solo en redes internas o con VPN.
- Errores del dashboard: mensajes amigables, sin stack traces expuestos.

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

# Activar entorno
source .venv/bin/activate

# Verificar
ruff check .
pytest
```

### Variables de entorno

Copia `.env.example` a `.env` y completa las credenciales necesarias (NASA Earthdata, Copernicus CDS, Google Earth Engine, etc.).

## Desarrollo

```bash
# Lint + format
ruff check . --fix
ruff format .

# Tests con cobertura
pytest

# Solo tests rápidos
pytest -m "not slow and not network"
```

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

## Licencia

MIT — ver `LICENSE`.

## Documentación adicional

- [`docs/prd.md`](docs/prd.md) — Product Requirements Document completo
- [`docs/data_dictionary.md`](docs/data_dictionary.md) — Diccionario de datos
- [`docs/model_cards/`](docs/model_cards/) — Tarjetas de modelo (M1, M2, M3)

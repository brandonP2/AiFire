# Handoff — petenfire2

**Fecha:** 2026-05-18  
**Directorio:** `/Users/mws/Documents/proyectos/AiFire/petenfire2`

---

## Contexto del proyecto

petenfire2 es la versión 2 de petenfire: predicción de riesgo de incendios forestales en el Petén, Guatemala. Adopta la metodología de la plantilla `model-template-build-uv` (8 notebooks + MLflow + GenericTransformer + config.yaml centralizado).

**Diferencia con v1:** Stacking ensemble (LightGBM + XGBoost + CatBoost + RF) con features espaciales nuevas. El modelo v1 tenía AUC-ROC 0.888 pero AUC-PR 0.007 (no detectaba fuegos reales).

---

## Estado actual del pipeline de 8 notebooks

| # | Notebook | Estado |
|---|----------|--------|
| 1 | `step1_load_data.ipynb` | **COMPLETO y ejecutado** |
| 2 | `step2_exploration.ipynb` | **COMPLETO y ejecutado** |
| 3 | `step3_preprocess.ipynb` | **INCOMPLETO — se estaba reescribiendo** |
| 4 | `step4_exploration_transformed.ipynb` | Pendiente |
| 5 | `step5_feature_selection.ipynb` | Pendiente |
| 6 | `step6_model_selection.ipynb` | Pendiente |
| 7 | `step7_train.ipynb` | Pendiente |
| 8 | `step8_register.ipynb` | Pendiente |

---

## Hallazgos clave del EDA (steps 1 y 2)

- **Dataset:** 136M filas, 28 columnas, 2018-2024
- **Target:** `fire_occurred` (bool) — 87,659 positivos (0.06%)
- **Desbalance:** 1551:1 → `scale_pos_weight = 1551` para base learners
- **Estacionalidad:** Abril (36%) + Mayo (34%) = 71% de todos los fuegos
- **Features más discriminativas:** `fwi` (ratio 3.9×), `isi_val` (3.2×), `RH2M` (correlación negativa)
- **Nulos:** solo NDVI — `ndvi` 0.40%, `ndvi_lag7`/`ndvi_lag14` 0.40%
- **Outliers para truncar:** `PRECTOTCORR`, `prec_acc7d`, `elevation_m`, `isi_val`, `fwi`
- **Columna `split`** ya existe en el dataset (train/val/test) — usarla directamente en Step 3
- **Features fire_lag y spatial NO existen** en el dataset — se construyen en Step 5

---

## Lo que se hizo en la última sesión antes de parar

Se estaba reescribiendo `step3_preprocess.ipynb`. El problema que se encontró:

1. **`transformations.json` tenía nombres incorrectos** (columnas de un dataset genérico, no del dataset real de petenfire2). Ya fue corregido — el archivo ahora tiene los nombres reales del schema.

2. **El notebook step3 original carga todo en RAM** (`pd.read_parquet` de 136M filas) → OOM garantizado.

3. La sesión se interrumpió justo antes de reescribir el notebook con la estrategia correcta.

---

## Lo que falta hacer en Step 3

### Estrategia para step3 (sin OOM)

El dataset es 3 GB y 136M filas. No cabe en RAM. La estrategia:

1. **Fit del GenericTransformer:** Cargar solo una muestra del train (~2M filas con todos los positivos + muestra de negativos). Las estadísticas de median/std no cambian significativamente con una muestra grande.

2. **Transform + guardado:** Procesar el dataset completo por batches con `pq.ParquetFile.iter_batches()` y escribir cada split en su archivo usando `pq.ParquetWriter`.

3. **Split:** La columna `split` ya existe — filtrar `split == 'train'`, `split == 'val'`, `split == 'test'` directamente.

### Pasos concretos del step3

```python
# 1. Muestra del train para fit
# Leer por batches filtrando split='train', tomar todos los positivos + N negativos

# 2. Preparar muestra: convertir is_protected_area a int, dropear [date, cell_id, split]

# 3. Fit del transformer en la muestra del train (sin el target)

# 4. Serializar transformer a data/processed/transformer.joblib

# 5. Transform completo por batches — para cada split:
#    pq.ParquetWriter → iter_batches → preprocesar cada batch → escribir
#    Output: data/processed/train.parquet, val.parquet, test.parquet

# 6. Verificar shapes y % de positivos en cada split
```

### Columnas a dropear en step3
- `date`, `cell_id`, `split` (definidas en `config.yaml` + `split`)
- `year`, `month`, `day_of_year` — son features temporales, mantener

### Conversión booleana
- `is_protected_area` (bool → int) antes de pasar al transformer

---

## Features pendientes de implementar (Step 5)

Estas features NO existen en el dataset actual y se deben construir:

| Feature | Descripción | Archivo |
|---------|-------------|---------|
| `fire_lag_1d` | Fuego en esta celda hace 1 día | `src/features/fire_lag_features.py` |
| `fire_lag_3d` | Fuego en esta celda hace 3 días | `src/features/fire_lag_features.py` |
| `fire_lag_7d` | Fuego en esta celda hace 7 días | `src/features/fire_lag_features.py` |
| `fire_lag_14d` | Fuego en esta celda hace 14 días | `src/features/fire_lag_features.py` |
| `fire_neighbors_3x3_7d` | Fuegos en vecindario 3×3 últimos 7 días | `src/features/spatial_features.py` |
| `fire_neighbors_5x5_7d` | Fuegos en vecindario 5×5 últimos 7 días | `src/features/spatial_features.py` |

El `transformations.json` ya tiene entradas para estas features (con `replace_nan: 0`). Cuando existan en el dataset, el transformer las procesará automáticamente.

---

## Infraestructura ya lista

- `src/models/stacking/base_learners.py` — 4 base learners implementados
- `src/models/stacking/temporal_split.py` — split temporal por año
- `src/models/stacking/oof.py` — OOF con folds anuales
- `src/models/stacking/meta_learner.py` — LogReg + calibración isotónica
- `src/models/stacking/ensemble.py` — `StackingEnsemble` orquestador
- `src/config/transformations.json` — corregido con nombres reales
- `src/config/config.yaml` — `objective_column: fire_occurred`, `scale_pos_weight: 1551`
- `data/raw/features.parquet` — symlink al parquet de v1 (3 GB)

---

## Estado real de los archivos de notebooks

Todos los 8 archivos existen en `notebooks/`. Los que no están completados son plantillas vacías:

| Notebook | Archivo existe | Ejecutado |
|----------|---------------|-----------|
| step1_load_data.ipynb | ✅ | ✅ |
| step2_exploration.ipynb | ✅ | ✅ |
| step3_preprocess.ipynb | ✅ | ❌ — reescribir desde cero |
| step4 → step8 | ✅ (plantillas) | ❌ — pendientes |

---

## Contenido de `transformations.json` (ya correcto)

El archivo `src/config/transformations.json` tiene los 26 campos procesados por GenericTransformer:

- **Clima/FWI con truncate + impute:** `T2M`, `RH2M`, `WS10M`, `PRECTOTCORR`, `prec_acc7d`, `prec_acc14d`, `fwi`, `ffmc_val`, `dmc_val`, `dc_val`, `isi_val`, `bui_val`
- **Topografía con truncate + impute:** `elevation_m`, `slope_deg`, `aspect_deg`, `dist_roads_km`, `dist_settlements_km`
- **NDVI:** `ndvi` (impute median + min_max[-1,1]), `ndvi_lag7` / `ndvi_lag14` (replace_nan 0 + min_max)
- **Fire features (pendientes Step 5):** `fire_lag_1d/3d/7d/14d`, `fire_neighbors_3x3_7d`, `fire_neighbors_5x5_7d` — todos con `replace_nan: 0`

**Columnas que NO están en transformations.json** (se manejan aparte):
- `is_protected_area` — convertir bool→int antes del transformer (no necesita normalización)
- `month`, `day_of_year`, `year` — features temporales, pasarlas al transformer sin transformación (el GenericTransformer las pasa tal cual si no están en el JSON)
- `cell_id`, `date`, `split` — se dropean antes de cualquier transformación

---

## Errores críticos corregidos en esta sesión

Estos errores ya están corregidos — NO revertir:

1. **`config.yaml`:** `objective_column` era `fire_label` → corregido a `fire_occurred`
2. **`transformations.json`:** tenía nombres genéricos (`temp_max`, `rh_mean`) → reescrito con los 28 nombres reales del schema
3. **`pyproject.toml`:** `requires-python` era `>=3.12` → cambiado a `>=3.11,<3.14` (catboost no soporta 3.14)
4. **`.python-version`:** creado con `3.11.9` para forzar Python 3.11 en uv
5. **symlink `data/`:** apunta a `../petenfire/data` (el dataset de v1)

---

## Para retomar la sesión

```bash
cd /Users/mws/Documents/proyectos/AiFire/petenfire2

# Verificar entorno (debe imprimir OK)
uv run python -c "import lightgbm, xgboost, catboost; print('OK')"

# Verificar dataset accesible (debe imprimir ~3.19 GB)
uv run python -c "import pyarrow.parquet as pq; pf = pq.ParquetFile('data/raw/features.parquet'); print(pf.metadata.num_rows)"

# Levantar MLflow (terminal separada, mantener abierta)
uv run mlflow ui --port 5000

# Abrir Jupyter
uv run jupyter lab notebooks/
```

**Primer paso al retomar:** reescribir y ejecutar `step3_preprocess.ipynb` siguiendo la estrategia de batches descrita arriba.

---

*Handoff generado: 2026-05-18. Sesión interrumpida justo antes de reescribir step3.*

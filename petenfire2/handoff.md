# Handoff — petenfire2

**Fecha:** 2026-05-18  
**Directorio:** `/Users/mws/Documents/proyectos/AiFire/petenfire/petenfire2` (dentro del repo)  
**Rama git:** `petenfire2`

---

## Contexto del proyecto

petenfire2 es la versión 2 de petenfire: predicción de riesgo de incendios forestales en el Petén, Guatemala. Adopta la metodología de la plantilla `model-template-build-uv` (8 notebooks + MLflow + GenericTransformer + config.yaml centralizado).

**Diferencia con v1:** Stacking ensemble (LightGBM + XGBoost + CatBoost + RF) con features espaciales nuevas. El modelo v1 tenía AUC-ROC 0.888 pero AUC-PR 0.007 (no detectaba fuegos reales).

---

## Estado actual del pipeline de 8 notebooks

| # | Notebook | Estado |
|---|----------|--------|
| 1 | `step1_load_data.ipynb` | **COMPLETO** |
| 2 | `step2_exploration.ipynb` | **COMPLETO** |
| 3 | `step3_preprocess.ipynb` | **COMPLETO** |
| 4 | `step4_exploration_transformed.ipynb` | **COMPLETO** |
| 5 | `step5_feature_selection.ipynb` | **PENDIENTE** ← siguiente |
| 6 | `step6_model_selection.ipynb` | Pendiente |
| 7 | `step7_train.ipynb` | Pendiente |
| 8 | `step8_register.ipynb` | Pendiente |

---

## Archivos generados por Step 3

| Archivo | Filas | Positivos | Tamaño |
|---------|-------|-----------|--------|
| `data/processed/train.parquet` | 97,165,112 | 54,162 | 2.3 GB |
| `data/processed/val.parquet` | 19,422,380 | 16,765 | 453 MB |
| `data/processed/test.parquet` | 19,475,592 | 16,732 | 410 MB |
| `data/processed/transformer.joblib` | — | — | ~1 KB |

**Nota:** Los parquets contienen 24 features transformadas + `fire_occurred` (target).

---

## Hallazgos clave del EDA (steps 1 y 2)

- **Dataset:** 136M filas, 28 columnas, 2018-2024
- **Target:** `fire_occurred` (bool) — 87,659 positivos (0.06%)
- **Desbalance:** 1551:1 → `scale_pos_weight = 1551` para base learners
- **Estacionalidad:** Abril (36%) + Mayo (34%) = 71% de todos los fuegos
- **Features más discriminativas:** `fwi` (ratio 3.9×), `isi_val` (3.2×), `RH2M` (correlación negativa)
- **Nulos:** solo NDVI — `ndvi` 0.40%, `ndvi_lag7`/`ndvi_lag14` 0.40%
- **Outliers truncados:** `PRECTOTCORR`, `prec_acc7d`, `elevation_m`, `isi_val`, `fwi`

---

## Features pendientes de implementar (Step 5)

Estas features NO existen en el dataset actual y se deben construir:

| Feature | Descripción |
|---------|-------------|
| `fire_lag_1d` | Fuego en esta celda hace 1 día |
| `fire_lag_3d` | Fuego en esta celda hace 3 días |
| `fire_lag_7d` | Fuego en esta celda hace 7 días |
| `fire_lag_14d` | Fuego en esta celda hace 14 días |
| `fire_neighbors_3x3_7d` | Fuegos en vecindario 3×3 últimos 7 días |
| `fire_neighbors_5x5_7d` | Fuegos en vecindario 5×5 últimos 7 días |

El `transformations.json` ya tiene entradas para estas features. Cuando existan, el transformer las procesará.

---

## Infraestructura ya lista

- `src/models/stacking/base_learners.py` — 4 base learners implementados
- `src/models/stacking/temporal_split.py` — split temporal por año
- `src/models/stacking/oof.py` — OOF con folds anuales
- `src/models/stacking/meta_learner.py` — LogReg + calibración isotónica
- `src/models/stacking/ensemble.py` — `StackingEnsemble` orquestador
- `src/config/transformations.json` — corregido con nombres reales
- `src/config/config.yaml` — `objective_column: fire_occurred`, `scale_pos_weight: 1551`
- `data/processed/*.parquet` — splits preprocesados listos para entrenar

---

## Para retomar la sesión

```bash
cd /Users/mws/Documents/proyectos/AiFire/petenfire/petenfire2

# Verificar entorno
uv run python -c "import lightgbm, xgboost, catboost; print('OK')"

# Verificar datos procesados
uv run python -c "import pyarrow.parquet as pq; print(pq.read_metadata('data/processed/train.parquet').num_rows)"

# Levantar MLflow (terminal separada)
uv run mlflow ui --port 5000

# Abrir Jupyter
uv run jupyter lab notebooks/
```

**Próximo paso:** Ejecutar `step5_feature_selection.ipynb` para ranking y selección de features.

---

## Hallazgos de Step 4 (validación post-transform)

- **Sin nulos** en ningún split
- **Rangos razonables** para todas las features
- **Top correlaciones con target:**
  - `year` (+0.68) — 2023/2024 tienen más fuegos proporcionalmente
  - `RH2M` (-0.55) — humedad negativa (esperado)
  - `dist_settlements_km` (-0.54) — más fuegos cerca de asentamientos
  - `T2M` (+0.49) — temperatura positiva (esperado)
  - `isi_val` (+0.48), `fwi` (+0.47) — índices de riesgo de fuego (esperado)
- **Balance preservado** entre splits (~0.06% positivos en cada uno)

---

## Commits en rama petenfire2

1. `fa55cc7` — feat(petenfire2): Initial commit
2. `f0e1f75` — feat(petenfire2): Complete step3 preprocessing
3. `ab19fc0` — docs(petenfire2): Update handoff after step3
4. `1459678` — feat(petenfire2): Complete step4 validation

---

*Handoff actualizado: 2026-05-18. Steps 1-4 completados.*

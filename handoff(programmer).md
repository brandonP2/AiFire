# Handoff — Programador

Fecha: 2026-05-11

---

## Meta actual

Obtener métricas reales (F1 > 0, AUC-PR > 0.01) del modelo **M1 LightGBM** entrenado con el dataset completo **2018–2024**. El modelo actual fue entrenado solo con 2022–2024 y tiene F1=0.0 porque el threshold de clasificación era 0.5 pero las probabilidades predichas son todas < 0.001.

---

## Estado actual del proyecto

### Lo que está listo
- **Grilla del Petén:** `data/interim/peten_grid.gpkg` — 53,212 celdas de 1 km²
- **Features estáticas:** `data/processed/static_features.parquet` — elevación, pendiente, distancias OSM, áreas protegidas
- **Clima NASA POWER 2018–2024:** `data/raw/power/power_{year}.parquet` — 7 años completos
- **Features climáticas + FWI 2018–2024:** `data/interim/climate/climate_features_{year}.parquet` — 7 años, ~96 MB/año
- **Etiquetas FIRMS 2018–2024:** `data/interim/fire_labels/fire_labels_{year}.parquet` — 7 años, 87,659 focos totales
- **FIRMS raw:** `data/raw/firms/` — VIIRS 2018–2021 + VIIRS 2022–2024 + MODIS 2022–2024. **Falta MODIS 2018–2021.**
- **Dashboard Streamlit:** `dashboard/app.py` corriendo en `http://localhost:8501` con 3 tabs (mapa de riesgo, métricas, análisis FIRMS)
- **Documentación:** `README.md`, `docs/data_dictionary.md`, `docs/prd.md` actualizados

### Lo que está roto / pendiente
- **`data/processed/m1_dataset.parquet` no existe** — fue borrado intencionalmente para forzar regeneración con datos 2018–2024. Hay que regenerarlo.
- **Modelo guardado es el viejo** (`models_artifacts/m1/`) — entrenado con 2022–2024, métricas: AUC-ROC=0.85, F1=0.0
- **FIRMS MODIS 2018–2021 nunca descargó** — solo existe `firms_VIIRS_SNPP_SP_2018-01_to_2021-12.csv`
- **NDVI es 100% NaN** — `data/interim/ndvi/` vacío, descarga de MOD13Q1 pendiente

---

## Todo lo que se intentó y falló

### Intento 1 — Pipeline watcher encadenado en background
Se lanzaron 3 procesos en background: `download_power`, `download_firms` (VIIRS + MODIS) para 2018–2021, y un watcher que esperaba el archivo FIRMS para lanzar `climate_features → fire_labels → build_dataset → train`.

**Falló porque:** el watcher esperaba `firms_VIIRS_SNPP_SP_2018-01_to_2021-12.csv`, el archivo apareció, pero el dataset que construyó era el de 2022–2024 (el parquet ya existía y el script lo saltó). El modelo se re-entrenó sobre los datos viejos.

### Intento 2 — Re-entrenamiento directo (sin regenerar dataset)
Se lanzó `train.py` directamente. Las métricas salieron idénticas al modelo anterior hasta el último decimal: AUC-ROC=0.8503551453606479.

**Falló porque:** el dataset era el mismo de 2022–2024. El entrenamiento no tocó los datos nuevos.

### Intento 3 — Pipeline encadenado en background (timeout)
Se borró el parquet y se lanzó `build_dataset → train` encadenados con `&&` en un solo comando background con timeout=600000ms.

**Falló porque:** el `build_dataset` con 7 años tarda ~15 min y el proceso fue matado (`status: killed`) antes de terminar. El dataset quedó sin generar.

### Intento 4 — Métricas reportadas como "ya terminó"
El usuario ejecutó el train desde su terminal. Las métricas reportadas fueron idénticas (AUC-ROC=0.8503551453606479, F1=0.0).

**Falló porque:** el `m1_dataset.parquet` no existía (había sido borrado), así que el train probablemente falló silenciosamente o encontró un dataset vacío. Además, el threshold del CLI seguía en 0.5 — bug adicional descubierto en ese momento.

---

## Cambios ya aplicados en `src/models/m1_risk/train.py`

Los tres cambios están en el archivo, **pero el modelo aún no fue reentrenado con ellos**:

| Línea | Cambio | Antes | Después |
|-------|--------|-------|---------|
| 312 | `scale_pos_weight` en `train_lightgbm` | `_compute_scale_pos_weight(y_train)` | `1.0  # SMOTE ya balanceó las clases` |
| 327 | `eval_metric` en `train_lightgbm` | `"auc"` | `"average_precision"` |
| 329 | callbacks en `train_lightgbm` | `early_stopping(50)`, `log_evaluation(100)` | `early_stopping(30)`, `log_evaluation(50)` |
| 382 | `threshold` default en `train_m1` | `0.5` | `0.05` |
| 532 | `threshold` default en CLI Typer | `0.5` | `0.05` |

---

## Siguiente paso

Ejecutar en la terminal, **en orden**:

```bash
cd /Users/mws/Desktop/Ai\ Fires/petenfire

# 1. Regenerar dataset completo 2018-2024 (~15 min)
uv run python -m src.features.build_dataset --start 2018-01-01 --end 2024-12-31

# 2. Verificar que tiene los 7 años antes de entrenar
uv run python -c "
import pandas as pd
df = pd.read_parquet('data/processed/m1_dataset.parquet', columns=['year'])
print(df['year'].value_counts().sort_index())
"
# Debe mostrar años 2018-2024 con ~19M filas cada uno

# 3. Entrenar con los cambios aplicados
uv run python -m src.models.m1_risk.train --model lightgbm --n-estimators 500

# 4. Ver métricas
cat models_artifacts/m1/metrics.yaml
```

Con los 7 años de datos y `threshold=0.05` debería aparecer F1 > 0. Si F1 sigue en 0.0, el siguiente diagnóstico es imprimir la distribución de `risk_prob` en val/test para ver si las probabilidades siguen siendo < 0.05.

---

## Archivos activamente editados

| Archivo | Qué se cambió |
|---------|--------------|
| `src/models/m1_risk/train.py` | 5 cambios de parámetros (scale_pos_weight, eval_metric, callbacks, threshold x2) — **cambios aplicados, modelo aún no reentrenado** |
| `src/models/m1_risk/__init__.py` | `CalibratedModel` extraído aquí desde `train.py` para que joblib serialice con módulo estable |
| `dashboard/app.py` | Dashboard Streamlit nuevo — mapa de riesgo + métricas + análisis FIRMS |
| `dashboard/utils.py` | Utilidades del dashboard — pushdown filter PyArrow, patch `__main__._CalibratedModel`, `load_model_feature_cols()` |
| `README.md` | Expandido con pipeline de datos, estado actual M1, decision log |
| `docs/data_dictionary.md` | Archivo nuevo — diccionario completo de todas las tablas y features |

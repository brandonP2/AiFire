# Handoff — PetenFire (Orquestador)

## Meta

Llevar el sistema PetenFire a **predecir incendios forestales en tiempo real** en el departamento del Petén, Guatemala:
1. M1 — Mapa de riesgo diario de ignición por celda de 1 km².
2. M2 — Simulación de propagación del fuego.
3. M3 — Detección visual de focos activos sobre imágenes Sentinel-2.

Desarrollo por fases: M1 → M2 → M3. Sin fecha límite, prioridad en calidad.

---

## Plan de orquestación M1

```
STAGE 1 — NDVI (bloqueante)
  ✅ 1A: Validar descarga (2024) — COMPLETADO
  ✅ 1B: Descarga completa 2018-2024 — COMPLETADO (184 tifs, 274 MB + 40 MB 2024 = ~314 MB total)
  ✅ 1C: Build NDVI features + Merge → m1_dataset.parquet — COMPLETADO (99.6% NDVI no-nulo)

STAGE 2 — Reentrenamiento en Colab
  ✅ 2A: Notebook anual base — COMPLETADO (AUC-ROC 0.888, AUC-PR 0.007, F1 0.023)
  ✅ 2B: Diagnóstico feature importance — COMPLETADO
  ❌ 2C: Modelo seasonal (mar-may) — DESCARTADO (AUC-ROC bajó a 0.752, best_iteration=1)
  ✅ 2D: Fire_lag features construidas — COMPLETADO (46 MB, 136M filas, anti-leakage OK)
  ✅ 2E: Notebook definitivo (anual + fire_lag) — COMPLETADO (notebooks/colab_train_m1_final.ipynb, 27 features, ~3GB RAM Colab)

STAGE 3 — Validación M1
  ⬜ AUC-PR > 0.10 | F1 > 0.20 | best_iteration > 1

STAGE 4 — Pipeline diario
  ⬜ Descarga automática clima + NDVI
  ⬜ Inferencia diaria sobre grilla Petén
  ⬜ Dashboard muestra mapa del día actual

### Lecciones aprendidas (no repetir)
- Estrategia seasonal EMPEORA el modelo: el contraste temporada/no-temporada era información valiosa
- early_stopping con average_precision en val sets pequeños dispara en iteración 1
- El gap real es ESPACIAL (no temporal): fire_lag features son el fix correcto
```

---

## Estado actual (2026-05-12)

### Stage 1A — COMPLETADO ✅

**Resultado:** Script GEE funciona correctamente.

- 23 GeoTIFFs descargados para 2024 en `data/interim/ndvi/`
- Tamaño: ~40 MB por año, ~280 MB total para 7 años
- CRS: EPSG:32616 ✓ | Shape: 877×1001 px a 250m ✓
- NDVI range: [-0.198, 0.999] float32, 0% NaN ✓

**Fix aplicado:** `GEE_PRIVATE_KEY_PATH` en `.env.local` tenía el path del proyecto anterior (Desktop). Corregido a la ruta actual.

**Script a usar:** `src/data/download_ndvi_gee.py` (GEE, no Earthdata)
- Razón: descarga solo el bbox del Petén (~2 MB/archivo vs 400 MB HDF globales)

---

### Lo que NO funciona aún

- **NDVI sin datos en el dataset:** las columnas `ndvi`, `ndvi_lag7`, `ndvi_lag14` son 100% NaN en `data/processed/m1_dataset.parquet` — Stage 1B/1C lo resuelve.
- **Modelo no detecta incendios:** AUC-PR = 0.004, F1 = 0.0 — se resuelve en Stage 2.
- **Sin pipeline diario:** no hay automatización — Stage 4.
- **M2 y M3:** módulos vacíos — fases posteriores.

---

## Decisiones de arquitectura

| Decisión | Elección | Razón |
|----------|----------|-------|
| Descarga NDVI | GEE (`download_ndvi_gee.py`) | 200× más pequeño que HDF globales |
| Compute entrenamiento | Google Colab Free | Mac se reinicia con 136M filas + SMOTE |
| Estrategia balanceo | Subsampleo 10:1 (no SMOTE) | SMOTE sobre 136M filas agota RAM |
| Validación temporal | train 2018-2022 / val 2023 / test 2024 | Series temporales — no K-fold aleatorio |

---

## Credenciales y paths críticos

| Recurso | Ubicación |
|---------|-----------|
| GEE service account JSON | `vast-bounty-495706-h0-4c1b23a8148f.json` (root del proyecto) |
| GEE_PRIVATE_KEY_PATH | `.env.local` — apunta al JSON anterior |
| Earthdata user/pass | `.env` — EARTHDATA_USER / EARTHDATA_PASS |
| FIRMS API key | `.env` — FIRMS_MAP_KEY |
| Dataset M1 | `data/processed/m1_dataset.parquet` (766 MB, 136M filas) |
| Modelo actual (roto) | `models_artifacts/m1/m1_lightgbm_calibrated.joblib` |

---

## Próximo comando exacto (Stage 1B)

```bash
uv run python -m src.data.download_ndvi_gee --start 2018 --end 2024
```

Tiempo estimado: ~5-6 minutos. Resultado esperado: ~161 GeoTIFFs en `data/interim/ndvi/`.

---

## Historial de intentos fallidos (no repetir)

### Intento 1 — Reentrenar con datos originales (2022-2024)
- Resultado: AUC-PR = 0.004, F1 = 0.0
- Causa: dataset cubría solo 3 años, insuficiente historia

### Intento 2 — Descargar FIRMS 2018-2021 y reentrenar
- Resultado: AUC-PR = 0.004, F1 = 0.0 (idéntico)
- Causa: bug en training — SMOTE + scale_pos_weight simultáneos (doble corrección), métrica early stopping `auc` inapropiada para clases tan raras, modelo se detiene en iteración 1

### Intento 3 — Entrenar localmente con dataset completo (136M filas)
- Resultado: Mac se reinicia por falta de RAM
- Causa: 136M filas + SMOTE requiere ~15-20 GB RAM

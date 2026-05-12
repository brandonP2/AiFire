# Handoff — PetenFire

## Meta

Llevar el sistema PetenFire a **predecir incendios forestales en tiempo real** en el departamento del Petén, Guatemala. Esto significa:
1. Un modelo M1 que produzca probabilidades de ignición útiles por celda de 1 km² cada día.
2. Un pipeline que descargue datos nuevos automáticamente cada día y regenere el mapa.
3. Un dashboard Streamlit que muestre el mapa del día actual sin intervención manual.

---

## Estado actual del proyecto (2026-05-11)

### Lo que ya funciona
- **Infraestructura completa:** grilla 1 km × 1 km del Petén construida, pipelines de descarga operativos.
- **Dataset M1:** 58 millones de filas (celdas × días, 2018–2024), con clima, FWI, topografía y etiquetas de fuego. Contiene **87,659 pares celda-día con fuego** confirmado.
- **Modelo LightGBM entrenado y guardado** en `models_artifacts/m1/m1_lightgbm_calibrated.joblib`.
- **Dashboard Streamlit funcional** con mapa de calor interactivo, superposición FIRMS y análisis de estacionalidad.
- **Tests unitarios** en verde.

### Lo que NO funciona aún
- **El modelo no detecta incendios reales:** AUC-PR = 0.004, F1 = 0.0. En la práctica no emitiría ninguna alerta útil.
- **NDVI sin datos:** las columnas `ndvi`, `ndvi_lag7`, `ndvi_lag14` son todas NaN porque los GeoTIFFs de MODIS/GEE no se descargaron.
- **M2 (propagación) y M3 (detección visual):** módulos vacíos, no implementados.
- **Sin pipeline de actualización diaria:** no hay automatización que descargue datos nuevos y regenere el mapa cada día.

---

## Todo lo que se intentó y falló

### Intento 1 — Reentrenar con datos originales (2022–2024)
- **Qué se hizo:** se entrenó LightGBM con el dataset que ya existía.
- **Resultado:** AUC-PR = 0.004, F1 = 0.0. Sin cambio respecto al modelo inicial.
- **Por qué falló:** el dataset solo cubría 2022–2024 y el entrenamiento no tenía suficiente historia de fuego.

### Intento 2 — Descargar FIRMS 2018–2021 y reentrenar
- **Qué se hizo:** se descargaron 70,000 focos históricos adicionales (tardó 6 horas), se reconstruyó el dataset y se reentrenó.
- **Resultado:** AUC-PR = 0.004, F1 = 0.0. Idéntico al intento anterior.
- **Por qué falló:** hay un bug en el código de entrenamiento — LightGBM se detiene en la iteración 1 porque se aplican SMOTE y `scale_pos_weight` simultáneamente (doble corrección del desbalance), y la métrica de early stopping (`auc`) no es apropiada para clases tan raras. El modelo aprende algo en la primera iteración y luego nunca mejora.

---

## Siguiente paso

Corregir tres líneas en `src/models/m1_risk/train.py` y reentrenar:

### Corrección 1 — Quitar la doble corrección de desbalance
En la función `train_lightgbm`, donde se calcula `scale_pos_weight`:
```python
# Línea actual (~312):
scale_pos_weight = _compute_scale_pos_weight(y_train)

# Cambiar a:
scale_pos_weight = 1.0  # SMOTE ya balanceó las clases
```

### Corrección 2 — Cambiar la métrica de early stopping
En la misma función, en el `model.fit`:
```python
# Línea actual (~328):
eval_metric="auc",

# Cambiar a:
eval_metric="average_precision",
```

### Corrección 3 — Bajar el umbral de clasificación
En la función `train_m1`, en la firma de la función:
```python
# Línea actual (~382):
threshold: float = 0.5,

# Cambiar a:
threshold: float = 0.05,
```

### Comando para reentrenar tras los cambios:
```bash
uv run python -m src.models.m1_risk.train
```

### Criterio de éxito:
- `auc_pr` en val > 0.10
- `f1` en val > 0.20
- `mejor iteración` debe ser > 1 (confirma que el modelo realmente entrenó)

Si las métricas pasan ese umbral, el siguiente paso es construir el pipeline de actualización diaria.

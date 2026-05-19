# CLAUDE.md - petenfire2

Instrucciones para Claude Code en este repositorio. **Este proyecto es una iteracion de petenfire v1** que adopta la metodologia de `model-template-build-uv` aplicada al problema de prediccion de incendios forestales en el Peten.

---

## 1. Contexto

**petenfire2** predice el riesgo diario de ignicion de incendios forestales en el **Peten, Guatemala** (~35,854 km^2).

- **Resolucion:** cuadricula 1 km x 1 km, prediccion diaria
- **Horizonte:** 1, 3 y 7 dias
- **Datos:** 2018-2024 (136M filas, ~53K celdas)
- **Split temporal:** train 2018-2022, val 2023, test 2024

### Diferencia con petenfire v1

| Aspecto | v1 | v2 (este repo) |
|---------|----|----------------|
| Modelo | LightGBM solo | Stacking ensemble (LGBM + XGB + CatBoost + RF) |
| Features espaciales | fire_lag basicos | + spatial aggregations 3x3, 5x5 (pendiente) |
| Balanceo | SMOTE (OOM) | class_weight + undersampling 10:1 |
| Calibracion | Isotonica post-hoc | Meta-learner LogReg + isotonic calibrado |
| Tracking | Manual | MLflow local |
| Metricas | AUC-PR 0.007 (fallido) | objetivo AUC-PR > 0.10, F1 > 0.20 |

---

## 2. Estructura del proyecto

Adoptamos la estructura de la plantilla `model-template-build-uv`:

```
petenfire2/
  src/
    project_config.py            # load_config(), init_notebook()
    analytics.py                 # threshold_iter, choose_best, train_model_from_df
    visualization/               # plots reutilizables (de plantilla)
    utils/                       # logger (modulo: 'petenfire2')
    transformers/                # GenericTransformer (de plantilla)
    config/
      config.yaml                # config principal
      transformations.json       # transformaciones por columna
    data/
      storage.py                 # LocalStorage (de plantilla)
      download_*.py              # downloaders heredados de v1
    features/                    # feature engineering heredado de v1
    models/
      stacking/
        base_learners.py         # LGBM, XGB, CatBoost, RF wrappers
        temporal_split.py        # split por año
        oof.py                   # OOF temporal por año
        meta_learner.py          # LogReg + calibracion isotonica
        ensemble.py              # StackingEnsemble
  notebooks/                     # 8 notebooks numerados step1..step8
  data/                          # symlink a ../petenfire/data (NO versionado)
  models_artifacts/              # joblibs finales
  docs/plans/                    # planes de implementacion
```

---

## 3. Pipeline de 8 notebooks

Ejecutar en orden:

| # | Notebook | Que hace |
|---|----------|----------|
| 1 | `step1_load_data.ipynb` | Symlink al `m1_dataset.parquet` de v1 (sin cargar 136M en RAM) |
| 2 | `step2_exploration.ipynb` | EDA del dataset crudo (muestra estratificada) |
| 3 | `step3_preprocess.ipynb` | GenericTransformer (fit en train) + temporal split |
| 4 | `step4_exploration_transformed.ipynb` | Validacion post-transformacion |
| 5 | `step5_feature_selection.ipynb` | Ranking LightGBM + seleccion top-N |
| 6 | `step6_model_selection.ipynb` | Base learners individuales en val 2023 |
| 7 | `step7_train.ipynb` | Stacking final + MLflow tracking |
| 8 | `step8_register.ipynb` | Registro en MLflow Model Registry |

Cada notebook inicia con:

```python
from project_config import init_notebook
config = init_notebook()
```

---

## 4. MLflow local

```bash
# Levantar MLflow UI local
mlflow ui --port 5000

# UI accesible en http://localhost:5000
```

Configurado en `src/config/config.yaml`:
- `tracking_uri: http://localhost:5000`
- `experiment_name: petenfire2-stacking`
- `registry_name: petenfire2-risk-classifier`

---

## 5. Comandos esenciales

```bash
# Setup
uv sync --extra dev

# Tests
uv run pytest

# Lint
uv run ruff check . --fix && uv run ruff format .

# MLflow UI (mantener abierta en otra terminal)
uv run mlflow ui --port 5000

# Jupyter
uv run jupyter lab notebooks/
```

---

## 6. Reglas heredadas de la plantilla

- **Temporal split SIEMPRE**: train 2018-2022, val 2023, test 2024. Nunca aleatorio.
- **Sin SMOTE**: `under_sampling_ratio=0.10` + `class_weight=balanced` (definido en `config.yaml`).
- **`threshold_strategy: max_recall`** con `min_precision=0.20` (falsos negativos cuestan mas).
- **Fit del transformer SOLO en train** para evitar leakage.
- **No usar `cell_id` ni `date` como features** (se dropean via `data.columns_to_drop`).
- **Funciones reutilizables** sin modificar: `threshold_iter`, `choose_best`, `evaluate_shap_values`.

---

## 7. Datos

Los datos de petenfire v1 son reutilizables. El Step 1 hace symlink al parquet maestro:

```bash
# Symlink a datos de v1 (recomendado)
ln -s ../petenfire/data data
```

---

## 8. Arquitectura del stacking

### Nivel 1 - Base learners (`src/models/stacking/base_learners.py`)

1. **LightGBM** - leaf-wise, rapido, `scale_pos_weight` + `metric=average_precision`
2. **XGBoost** - level-wise, `scale_pos_weight` + `eval_metric=aucpr`
3. **CatBoost** - `auto_class_weights=Balanced` + `eval_metric=PRAUC`
4. **RandomForest** - `class_weight=balanced`, sin early stopping

Todos exponen `fit(X, y, X_val, y_val)` y `get_feature_importance() -> dict`.

### OOF temporal (`src/models/stacking/oof.py`)

Folds anuales (no aleatorios):
- fold 0: train 2018, val 2019
- fold 1: train 2018-2019, val 2020
- ...

### Nivel 2 - Meta-learner (`src/models/stacking/meta_learner.py`)

- `LogisticRegression(class_weight='balanced')`
- Envuelta en `CalibratedClassifierCV(method='isotonic', cv=3)`
- Input: matriz [n_samples, 4] con las OOF de los base learners

### Orquestador (`src/models/stacking/ensemble.py`)

`StackingEnsemble.fit(X, y, dates, X_val, y_val)` ejecuta el pipeline completo y deja modelos reentrenados en todo el train.

---

## 9. Metricas objetivo

| Metrica | v1 | v2 objetivo |
|---------|----|-------------|
| AUC-ROC | 0.888 | > 0.85 |
| AUC-PR | 0.007 | > 0.10 |
| F1 | 0.023 | > 0.20 |
| Recall (high risk) | - | > 0.70 |

---

## 10. Reglas inviolables

- NUNCA commitear `data/`, `.env`, credenciales
- NUNCA usar K-fold aleatorio en series temporales
- NUNCA combinar SMOTE + scale_pos_weight
- Si dataset > 50M filas, usar pyarrow streaming o chunking
- Cada modelo entrenado debe quedar trackeado en MLflow

---

## 11. Plan de implementacion

Ver `docs/plans/01-stacking-ensemble.md` para detalles y checkpoints.

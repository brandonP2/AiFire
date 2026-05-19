# Plan 01 - Stacking Ensemble (metodologia plantilla ML)

Este plan adopta la metodologia de `model-template-build-uv` y la aplica al
problema de prediccion de riesgo de incendios forestales en el Peten.

## Resumen ejecutivo

Construimos un stacking ensemble (LightGBM + XGBoost + CatBoost + RandomForest)
con meta-learner LogisticRegression calibrado isotonicamente. El flujo se
ejecuta en 8 notebooks numerados, con MLflow tracking desde Step 7. El
preprocesamiento usa el `GenericTransformer` de la plantilla y un
`transformations.json` especifico de petenfire2. El split es estrictamente
temporal: train 2018-2022, val 2023, test 2024.

## Por que esta arquitectura

| Decision | Razon |
|----------|-------|
| Stacking de 4 modelos | v1 con un solo LightGBM dio AUC-PR 0.007. Diversidad compensa errores correlacionados. |
| Meta-learner calibrado | El threshold debe corresponder a probabilidades reales para `threshold_strategy: max_recall`. |
| Sin SMOTE | Dataset 136M filas - OOM. Usamos `class_weight=balanced` y `under_sampling_ratio=0.10`. |
| Split temporal estricto | Series de tiempo. K-fold aleatorio = leakage. |
| `max_recall` con `min_precision=0.20` | Costo de falso negativo supera al de falso positivo. |
| MLflow local | Reproducibilidad sin servidor remoto. |

## Pipeline de 8 notebooks

| Notebook | Proposito | Outputs |
|----------|-----------|---------|
| `step1_load_data.ipynb` | Symlink al `m1_dataset.parquet` de v1 (no carga 136M en RAM) | `data/raw/features.parquet` |
| `step2_exploration.ipynb` | EDA del dataset crudo (muestra estratificada) | Notebook |
| `step3_preprocess.ipynb` | `GenericTransformer` fit en train, transform en val/test + temporal split | `data/processed/{train,val,test}.parquet` + `transformer.joblib` |
| `step4_exploration_transformed.ipynb` | Validacion post-transformacion | Notebook |
| `step5_feature_selection.ipynb` | LightGBM rapido para ranking, seleccion top-40 respetando `fixed_features` | `selected_features.json` |
| `step6_model_selection.ipynb` | Evaluacion individual de cada base learner en val 2023 | Tabla comparativa |
| `step7_train.ipynb` | Stacking completo + MLflow tracking | `stacking_ensemble.joblib` + run en MLflow |
| `step8_register.ipynb` | Promocion a MLflow Model Registry | Version registrada |

## Checkpoints

| Checkpoint | Criterio go/no-go |
|------------|-------------------|
| Step 3 | Train >= 5 años, sin NaN. Val 2023 representa >10% del train. |
| Step 5 | Top-40 incluye `fwi`, `ndvi` y al menos un `fire_lag_*`. |
| Step 6 | Cada base learner individual alcanza PR-AUC > 0.05 en val. |
| Step 7 | Ensemble val PR-AUC > 0.10. Si no, iterar features espaciales. |
| Step 8 | Modelo registrado, threshold guardado, comparativa con v1 en MLflow. |

## Estructura del codigo

```
petenfire2/
  src/
    project_config.py            # load_config(), init_notebook()
    analytics.py                 # threshold_iter, choose_best, train_model_from_df (sin SMOTE)
    visualization/               # plots reutilizables
    utils/                       # logger
    transformers/                # GenericTransformer
    data/
      storage.py                 # LocalStorage
      ...                        # downloaders heredados de v1
    config/
      config.yaml                # config petenfire2
      transformations.json       # transformaciones por columna
    models/
      stacking/
        base_learners.py         # LGBM, XGB, CatBoost, RF wrappers
        temporal_split.py        # split por año
        oof.py                   # OOF temporal por año
        meta_learner.py          # LogReg + calibracion isotonica
        ensemble.py              # StackingEnsemble
  notebooks/                     # 8 notebooks numerados
  models_artifacts/              # joblib finales
```

## Reglas inviolables

1. **Temporal split SIEMPRE**: train 2018-2022, val 2023, test 2024.
2. **Sin SMOTE**: `under_sampling_ratio=0.10` + `class_weight=balanced`.
3. **Threshold via estrategia**: `max_recall` con `min_precision=0.20`.
4. **MLflow local** (`http://localhost:5000`) durante desarrollo.
5. **Fit del transformer SOLO en train**.
6. **Sin `cell_id` ni `date` como features**.

## Tareas pendientes despues del MVP

- Spatial aggregations (vecindario 3x3, 5x5) reutilizando downloaders de v1.
- Hot cluster features (distancia al cluster mas cercano).
- Dashboard streamlit para servir predicciones del ensemble.
- CI con `pytest` para `analytics.py` y `models/stacking/`.

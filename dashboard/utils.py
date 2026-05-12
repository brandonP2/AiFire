"""Utilidades de carga y caché para el dashboard PetenFire."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import geopandas as gpd
import joblib
import pandas as pd
import streamlit as st
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_RAW = PROJECT_ROOT / "data" / "raw"
MODEL_DIR = PROJECT_ROOT / "models_artifacts" / "m1"


@st.cache_resource
def load_model():
    model_path = MODEL_DIR / "m1_lightgbm_calibrated.joblib"
    if not model_path.exists():
        return None
    import sys
    from src.models.m1_risk import CalibratedModel
    # El modelo actual fue serializado con _CalibratedModel en __main__.
    # Registrar ambos nombres para que joblib pueda deserializarlo.
    sys.modules["__main__"]._CalibratedModel = CalibratedModel
    sys.modules["__main__"].CalibratedModel = CalibratedModel
    return joblib.load(model_path)


@st.cache_data
def load_grid() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(DATA_INTERIM / "peten_grid.gpkg")
    return gdf.to_crs("EPSG:4326")


@st.cache_data
def load_dataset_dates() -> list[date]:
    import pyarrow.parquet as pq
    pf = pq.read_table(DATA_PROCESSED / "m1_dataset.parquet", columns=["date"])
    dates = pf["date"].to_pandas()
    dates = pd.to_datetime(dates).dt.date
    return sorted(dates.unique().tolist())


@st.cache_data
def load_features_for_date(target_date: date) -> pd.DataFrame:
    from src.models.m1_risk.train import FEATURE_COLS
    import numpy as np

    dataset_path = DATA_PROCESSED / "m1_dataset.parquet"
    # Pushdown filter: lee solo las filas del día solicitado
    df = pd.read_parquet(
        dataset_path,
        filters=[("date", "=", target_date)],
    )
    if df.empty:
        raise ValueError(f"No hay datos para {target_date}")
    df["date"] = pd.to_datetime(df["date"]).dt.date

    available = [c for c in FEATURE_COLS if c in df.columns]
    result = df[["cell_id", *available]].copy()

    for col in available:
        if result[col].isna().any():
            result[col] = result[col].fillna(result[col].median())

    return result


@st.cache_data
def load_firms() -> pd.DataFrame:
    dfs = []
    for f in (DATA_RAW / "firms").glob("*.csv"):
        df = pd.read_csv(f, usecols=["latitude", "longitude", "acq_date", "confidence", "frp", "source"])
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    firms = pd.concat(dfs, ignore_index=True)
    firms["acq_date"] = pd.to_datetime(firms["acq_date"]).dt.date
    return firms


@st.cache_data
def load_metrics() -> dict:
    metrics_path = MODEL_DIR / "metrics.yaml"
    if not metrics_path.exists():
        return {}
    with open(metrics_path) as f:
        return yaml.safe_load(f)


@st.cache_data
def load_experiment_config() -> dict:
    cfg_path = MODEL_DIR / "experiment_config.yaml"
    if not cfg_path.exists():
        return {}
    with open(cfg_path) as f:
        return yaml.safe_load(f)


@st.cache_data
def load_model_feature_cols() -> list[str]:
    """Devuelve las features con las que fue entrenado el modelo actual."""
    cfg = load_experiment_config()
    if cfg and "features" in cfg:
        return cfg["features"]
    # Fallback: todas las features del train.py
    from src.models.m1_risk.train import FEATURE_COLS
    return FEATURE_COLS


def get_feature_importance() -> pd.DataFrame | None:
    """Extrae feature importance del modelo LightGBM base."""
    model = load_model()
    if model is None:
        return None
    base = getattr(model, "_base", model)
    if not hasattr(base, "feature_importances_"):
        return None
    feature_cols = load_model_feature_cols()
    imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": base.feature_importances_,
    }).sort_values("importance", ascending=False)
    return imp

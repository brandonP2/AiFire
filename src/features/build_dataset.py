"""Pipeline principal de feature engineering para el modelo M1 de riesgo de incendio.

Orquesta los módulos:
    static_features   → topografía, distancias, áreas protegidas (por celda)
    climate_features  → NASA POWER + FWI iterativo (por celda/día)
    ndvi_features     → NDVI diario interpolado + lags (por celda/día)
    fire_labels       → etiqueta fire_occurred de NASA FIRMS (por celda/día)

Salida final:
    data/processed/m1_dataset.parquet — una fila por (cell_id, date)
    Columnas: todas las definidas en src.data.schemas.CellDayRecord

Uso:
    uv run python -m src.features.build_dataset
    uv run python -m src.features.build_dataset --start 2018-01-01 --end 2024-12-31
    uv run python -m src.features.build_dataset --skip-existing
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Annotated, Literal

import pandas as pd
import typer
from loguru import logger

from src.data.config import Settings
from src.data.config import settings as default_settings
from src.features.climate_features import build_climate_features
from src.features.fire_labels import build_fire_labels
from src.features.ndvi_features import build_ndvi_features
from src.features.static_features import build_static_features

app = typer.Typer(add_completion=False)

# ---------------------------------------------------------------------------
# Split temporal
# ---------------------------------------------------------------------------

_SPLIT_RULES: list[tuple[str, str, Literal["train", "val", "test"]]] = [
    ("2018-01-01", "2022-12-31", "train"),
    ("2023-01-01", "2023-12-31", "val"),
    ("2024-01-01", "2024-12-31", "test"),
]


def _assign_split(dates: pd.Series) -> pd.Series:
    """Asigna la partición temporal a cada fecha.

    Args:
        dates: Series de objetos date.

    Returns:
        Series con valores 'train', 'val' o 'test'.
    """
    split = pd.Series("train", index=dates.index, dtype=str)
    for start_s, end_s, label in _SPLIT_RULES:
        s = date.fromisoformat(start_s)
        e = date.fromisoformat(end_s)
        mask = dates.apply(lambda d, _s=s, _e=e: _s <= d <= _e)
        split[mask] = label
    return split


# ---------------------------------------------------------------------------
# Precipitación acumulada
# ---------------------------------------------------------------------------


def _compute_prec_accumulados(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula precipitación acumulada a 7 y 14 días por celda.

    Usa pivot para operar como matriz (fechas × celdas) y aplicar rolling
    vectorizado — evita el loop implícito de groupby+transform sobre 136M filas.
    """
    df = df.sort_values(["cell_id", "date"]).reset_index(drop=True)

    pivot = df.pivot(index="date", columns="cell_id", values="PRECTOTCORR")
    acc7 = pivot.rolling(window=7, min_periods=1).sum()
    acc14 = pivot.rolling(window=14, min_periods=1).sum()

    acc7_flat = acc7.stack().rename("prec_acc7d")
    acc14_flat = acc14.stack().rename("prec_acc14d")
    acc_df = pd.concat([acc7_flat, acc14_flat], axis=1).reset_index()
    acc_df.columns = ["date", "cell_id", "prec_acc7d", "prec_acc14d"]

    df = df.merge(acc_df, on=["cell_id", "date"], how="left")
    return df


# ---------------------------------------------------------------------------
# Carga de parquets intermedios
# ---------------------------------------------------------------------------


def _load_parquets(directory: Path, pattern: str) -> pd.DataFrame | None:
    """Carga y concatena todos los parquets que coincidan con el patrón.

    Args:
        directory: Directorio donde buscar.
        pattern: Glob pattern (e.g. 'climate_features_*.parquet').

    Returns:
        DataFrame concatenado, o None si no hay archivos.
    """
    paths = sorted(directory.glob(pattern))
    if not paths:
        return None
    dfs = [pd.read_parquet(p) for p in paths]
    return pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------


def build_m1_dataset(
    cfg: Settings | None = None,
    start: date | None = None,
    end: date | None = None,
    grid_path: Path | None = None,
    output_path: Path | None = None,
    skip_existing: bool = False,
) -> pd.DataFrame:
    """Construye el dataset completo para M1.

    Ejecuta todos los pasos del pipeline en orden:
    1. Features estáticas (topografía, distancias, áreas protegidas).
    2. Features climáticas diarias y FWI.
    3. NDVI diario y lags.
    4. Etiquetas de incendio FIRMS.
    5. Ensamblado, precipitación acumulada y asignación de split.

    Args:
        cfg: Configuración del proyecto.
        start: Fecha de inicio del período.
        end: Fecha de fin del período.
        grid_path: Ruta al GeoPackage de la grilla.
        output_path: Ruta del parquet final.
        skip_existing: Si True, omite pasos cuyos outputs ya existen.

    Returns:
        DataFrame final del dataset M1.
    """
    cfg = cfg or default_settings
    start = start or date.fromisoformat(cfg.train_start)
    end = end or date.fromisoformat(cfg.test_end)
    grid_path = grid_path or (cfg.data_interim / "peten_grid.gpkg")
    output_path = output_path or (cfg.data_processed / "m1_dataset.parquet")

    if skip_existing and output_path.exists():
        logger.info(f"Dataset ya existe: {output_path} — cargando")
        return pd.read_parquet(output_path)

    logger.info("=" * 60)
    logger.info(f"Build M1 dataset: {start} → {end}")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Paso 1: Features estáticas
    # ------------------------------------------------------------------
    static_path = cfg.data_processed / "static_features.parquet"
    if skip_existing and static_path.exists():
        logger.info("Paso 1: Cargando features estáticas existentes...")
        static_df = pd.read_parquet(static_path)
    else:
        logger.info("Paso 1: Computando features estáticas...")
        static_df = build_static_features(cfg=cfg, grid_path=grid_path, output_path=static_path)

    # ------------------------------------------------------------------
    # Paso 2: Features climáticas + FWI
    # ------------------------------------------------------------------
    logger.info("Paso 2: Procesando clima y FWI...")
    build_climate_features(cfg=cfg, start=start, end=end, grid_path=grid_path)
    climate_df = _load_parquets(cfg.data_interim / "climate", "climate_features_*.parquet")
    if climate_df is None:
        raise RuntimeError(
            "No se generaron features climáticas. "
            "Verifica que existan archivos en data/raw/power/."
        )

    # Filtrar al rango solicitado
    climate_df["date"] = pd.to_datetime(climate_df["date"]).dt.date
    climate_df = climate_df[
        (climate_df["date"] >= start) & (climate_df["date"] <= end)
    ]
    logger.info(f"Clima: {len(climate_df):,} filas ({climate_df['date'].nunique()} días)")

    # ------------------------------------------------------------------
    # Paso 3: NDVI diario + lags
    # ------------------------------------------------------------------
    logger.info("Paso 3: Procesando NDVI...")
    build_ndvi_features(cfg=cfg, start=start, end=end, grid_path=grid_path)
    ndvi_df = _load_parquets(cfg.data_interim / "ndvi", "ndvi_features_*.parquet")

    if ndvi_df is not None:
        ndvi_df["date"] = pd.to_datetime(ndvi_df["date"]).dt.date
        ndvi_df = ndvi_df[
            (ndvi_df["date"] >= start) & (ndvi_df["date"] <= end)
        ]
        logger.info(f"NDVI: {len(ndvi_df):,} filas")
    else:
        logger.warning("Sin datos NDVI disponibles — columnas NDVI serán NaN")

    # ------------------------------------------------------------------
    # Paso 4: Etiquetas FIRMS
    # ------------------------------------------------------------------
    logger.info("Paso 4: Generando etiquetas de incendio...")
    build_fire_labels(cfg=cfg, start=start, end=end, grid_path=grid_path)
    fire_df = _load_parquets(cfg.data_interim / "fire_labels", "fire_labels_*.parquet")

    if fire_df is not None:
        fire_df["date"] = pd.to_datetime(fire_df["date"]).dt.date
        fire_df = fire_df[
            (fire_df["date"] >= start) & (fire_df["date"] <= end)
        ]
        logger.info(f"Incendios: {len(fire_df):,} pares (celda, día) con fuego")
    else:
        logger.warning("Sin datos FIRMS disponibles — fire_occurred será False por defecto")

    # ------------------------------------------------------------------
    # Paso 5: Ensamblado
    # ------------------------------------------------------------------
    logger.info("Paso 5: Ensamblando dataset final...")

    # Base: clima (cubre todas las celdas x todos los dias)
    df = climate_df.copy()

    # Unir NDVI
    if ndvi_df is not None:
        df = df.merge(
            ndvi_df[["cell_id", "date", "ndvi", "ndvi_lag7", "ndvi_lag14"]],
            on=["cell_id", "date"],
            how="left",
        )
    else:
        df["ndvi"] = float("nan")
        df["ndvi_lag7"] = float("nan")
        df["ndvi_lag14"] = float("nan")

    # Unir etiquetas de incendio
    if fire_df is not None:
        df = df.merge(
            fire_df[["cell_id", "date", "fire_occurred"]],
            on=["cell_id", "date"],
            how="left",
        )
        df["fire_occurred"] = df["fire_occurred"].fillna(False)
    else:
        df["fire_occurred"] = False

    # Unir features estáticas
    df = df.merge(static_df, on="cell_id", how="left")

    # ------------------------------------------------------------------
    # Paso 6: Features derivadas temporales
    # ------------------------------------------------------------------
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["year"] = df["date"].dt.year
    df["date"] = df["date"].dt.date

    # Precipitación acumulada
    df = _compute_prec_accumulados(df)

    # FWI ya viene calculado desde climate_features (columna fwi_val)
    if "fwi_val" in df.columns:
        df = df.rename(columns={"fwi_val": "fwi"})

    # Asignar split
    df["split"] = _assign_split(df["date"])

    # ------------------------------------------------------------------
    # Orden final de columnas (según CellDayRecord)
    # ------------------------------------------------------------------
    ordered_cols = [
        "cell_id", "date", "fire_occurred",
        # Clima
        "T2M", "RH2M", "WS10M", "PRECTOTCORR",
        # FWI
        "fwi", "ffmc_val", "dmc_val", "dc_val", "isi_val", "bui_val",
        # Precipitación acumulada
        "prec_acc7d", "prec_acc14d",
        # Vegetación
        "ndvi", "ndvi_lag7", "ndvi_lag14",
        # Topografía
        "elevation_m", "slope_deg", "aspect_deg",
        # Antrópico
        "dist_roads_km", "dist_settlements_km", "is_protected_area",
        # Temporal
        "month", "day_of_year", "year",
        # Split
        "split",
    ]
    # Incluir solo las columnas que existan
    final_cols = [c for c in ordered_cols if c in df.columns]
    # Añadir columnas extra que no estén en el orden definido
    extra_cols = [c for c in df.columns if c not in ordered_cols]
    df = df[final_cols + extra_cols]

    # ------------------------------------------------------------------
    # Guardar
    # ------------------------------------------------------------------
    n_rows = len(df)
    n_fire = df["fire_occurred"].sum()
    fire_rate = 100 * n_fire / n_rows if n_rows > 0 else 0.0

    logger.info(f"Dataset M1: {n_rows:,} filas x {df.shape[1]} columnas")
    logger.info(f"  Distribución de clases: {n_fire:,} fuego ({fire_rate:.2f}%), "
                f"{n_rows - n_fire:,} sin fuego")
    logger.info(f"  Splits: {df.groupby('split').size().to_dict()}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.success(f"Dataset guardado en {output_path}")

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    start: Annotated[
        str,
        typer.Option("--start", help="Fecha inicio (YYYY-MM-DD)"),
    ] = "2018-01-01",
    end: Annotated[
        str,
        typer.Option("--end", help="Fecha fin (YYYY-MM-DD)"),
    ] = "2024-12-31",
    grid_path: Annotated[
        Path,
        typer.Option("--grid-path", help="Ruta al GeoPackage de la grilla"),
    ] = Path("data/interim/peten_grid.gpkg"),
    output_path: Annotated[
        Path,
        typer.Option("--output-path", "-o", help="Ruta del parquet final"),
    ] = Path("data/processed/m1_dataset.parquet"),
    skip_existing: Annotated[
        bool,
        typer.Option("--skip-existing", help="Omitir pasos con outputs existentes"),
    ] = False,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="Nivel de logging"),
    ] = "INFO",
) -> None:
    """Construye el dataset completo para el modelo M1 de riesgo de incendio."""
    logger.remove()
    logger.add(sys.stderr, level=log_level.upper())

    build_m1_dataset(
        start=date.fromisoformat(start),
        end=date.fromisoformat(end),
        grid_path=grid_path,
        output_path=output_path,
        skip_existing=skip_existing,
    )


if __name__ == "__main__":
    app()

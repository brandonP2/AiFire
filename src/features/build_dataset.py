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
from datetime import date, timedelta
from pathlib import Path
from typing import Annotated, Literal

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
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


def _compute_prec_accumulados(
    df: pd.DataFrame,
    prefix_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Calcula precipitación acumulada a 7 y 14 días por celda.

    Usa pivot para operar como matriz (fechas × celdas) y aplicar rolling
    vectorizado. Cuando se proporciona prefix_df (últimos 14 días del año
    anterior), el rolling queda correctamente inicializado en el límite de año.

    Args:
        df: DataFrame del año a procesar (cell_id, date, PRECTOTCORR, …).
        prefix_df: Tail del año anterior con columnas (cell_id, date, PRECTOTCORR).

    Returns:
        df con columnas prec_acc7d y prec_acc14d añadidas.
    """
    prec_cols = ["cell_id", "date", "PRECTOTCORR"]
    if prefix_df is not None and not prefix_df.empty:
        combined = pd.concat(
            [prefix_df[prec_cols], df[prec_cols]], ignore_index=True
        )
    else:
        combined = df[prec_cols].copy()

    combined = combined.sort_values(["cell_id", "date"]).reset_index(drop=True)
    pivot = combined.pivot(index="date", columns="cell_id", values="PRECTOTCORR")
    acc7 = pivot.rolling(window=7, min_periods=1).sum()
    acc14 = pivot.rolling(window=14, min_periods=1).sum()

    # Conservar solo las fechas del año actual
    current_dates = df["date"].unique()
    acc7 = acc7.loc[acc7.index.isin(current_dates)]
    acc14 = acc14.loc[acc14.index.isin(current_dates)]

    acc7_flat = acc7.stack().rename("prec_acc7d").reset_index()
    acc14_flat = acc14.stack().rename("prec_acc14d").reset_index()
    acc7_flat.columns = ["date", "cell_id", "prec_acc7d"]
    acc14_flat.columns = ["date", "cell_id", "prec_acc14d"]
    acc_df = acc7_flat.merge(acc14_flat, on=["date", "cell_id"])

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
    # Paso 2: Asegurar datos intermedios generados
    # ------------------------------------------------------------------
    logger.info("Paso 2: Procesando clima y FWI...")
    build_climate_features(cfg=cfg, start=start, end=end, grid_path=grid_path)

    logger.info("Paso 3: Procesando NDVI (si hay GeoTIFFs disponibles)...")
    build_ndvi_features(cfg=cfg, start=start, end=end, grid_path=grid_path)

    logger.info("Paso 4: Generando etiquetas de incendio...")
    build_fire_labels(cfg=cfg, start=start, end=end, grid_path=grid_path)

    # ------------------------------------------------------------------
    # Paso 5: Ensamblado año por año (evita cargar 136M filas de golpe)
    # ------------------------------------------------------------------
    logger.info("Paso 5: Ensamblando dataset año por año...")

    ordered_cols = [
        "cell_id", "date", "fire_occurred",
        "T2M", "RH2M", "WS10M", "PRECTOTCORR",
        "fwi", "ffmc_val", "dmc_val", "dc_val", "isi_val", "bui_val",
        "prec_acc7d", "prec_acc14d",
        "ndvi", "ndvi_lag7", "ndvi_lag14",
        "elevation_m", "slope_deg", "aspect_deg",
        "dist_roads_km", "dist_settlements_km", "is_protected_area",
        "month", "day_of_year", "year",
        "split",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    total_rows = 0
    total_fire = 0

    years = list(range(start.year, end.year + 1))
    prev_climate_tail: pd.DataFrame | None = None  # últimos 14 días del año anterior

    for year in years:
        year_start = max(start, date(year, 1, 1))
        year_end = min(end, date(year, 12, 31))

        climate_path = cfg.data_interim / "climate" / f"climate_features_{year}.parquet"
        if not climate_path.exists():
            logger.warning(f"  {year}: sin datos de clima — saltando")
            prev_climate_tail = None
            continue

        logger.info(f"  Ensamblando {year}...")

        # Cargar clima del año y filtrar al rango
        climate_df = pd.read_parquet(climate_path)
        climate_df["date"] = pd.to_datetime(climate_df["date"]).dt.date
        climate_df = climate_df[
            (climate_df["date"] >= year_start) & (climate_df["date"] <= year_end)
        ]

        # Precipitación acumulada usando cola del año anterior para el rolling
        climate_df = _compute_prec_accumulados(climate_df, prefix_df=prev_climate_tail)

        # Guardar cola de este año para el siguiente (últimos 14 días)
        tail_cutoff = year_end - timedelta(days=14)
        prev_climate_tail = (
            pd.read_parquet(climate_path, columns=["cell_id", "date", "PRECTOTCORR"])
            .assign(date=lambda d: pd.to_datetime(d["date"]).dt.date)
            .pipe(lambda d: d[d["date"] >= tail_cutoff])
        )

        df = climate_df.copy()
        del climate_df

        # FWI: renombrar si viene como fwi_val
        if "fwi_val" in df.columns:
            df = df.rename(columns={"fwi_val": "fwi"})

        # NDVI (NaN hasta que haya GeoTIFFs descargados)
        df["ndvi"] = float("nan")
        df["ndvi_lag7"] = float("nan")
        df["ndvi_lag14"] = float("nan")

        # Etiquetas de incendio
        fire_path = cfg.data_interim / "fire_labels" / f"fire_labels_{year}.parquet"
        if fire_path.exists():
            fire_df = pd.read_parquet(fire_path)
            fire_df["date"] = pd.to_datetime(fire_df["date"]).dt.date
            df = df.merge(
                fire_df[["cell_id", "date", "fire_occurred"]],
                on=["cell_id", "date"],
                how="left",
            )
            df["fire_occurred"] = df["fire_occurred"].fillna(False)
        else:
            df["fire_occurred"] = False

        # Features estáticas
        df = df.merge(static_df, on="cell_id", how="left")

        # Features temporales
        df["date"] = pd.to_datetime(df["date"])
        df["month"] = df["date"].dt.month
        df["day_of_year"] = df["date"].dt.dayofyear
        df["year"] = df["date"].dt.year
        df["date"] = df["date"].dt.date

        # Split
        df["split"] = _assign_split(df["date"])

        # Orden final de columnas
        final_cols = [c for c in ordered_cols if c in df.columns]
        extra_cols = [c for c in df.columns if c not in ordered_cols]
        df = df[final_cols + extra_cols]

        # Escribir al parquet de salida (append con ParquetWriter)
        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")
        writer.write_table(table)

        n_fire_year = int(df["fire_occurred"].sum())
        total_rows += len(df)
        total_fire += n_fire_year
        logger.info(f"    {year}: {len(df):,} filas, {n_fire_year:,} focos de fuego")

        del df, table
        import gc
        gc.collect()

    if writer is not None:
        writer.close()

    fire_rate = 100 * total_fire / total_rows if total_rows > 0 else 0.0
    logger.info(f"Dataset M1: {total_rows:,} filas totales")
    logger.info(
        f"  Distribución: {total_fire:,} fuego ({fire_rate:.2f}%), "
        f"{total_rows - total_fire:,} sin fuego"
    )
    logger.success(f"Dataset guardado en {output_path}")

    return pd.read_parquet(output_path)


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

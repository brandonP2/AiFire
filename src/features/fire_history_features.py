"""Feature engineering de historial de fuego para el modelo M1.

Construye cuatro features de historial de incendios por celda-día:
    - fire_cell_lag30d:   ¿Ardió esta celda en los últimos 30 días?        (binario)
    - fire_cell_lag365d:  ¿Ardió esta celda en los últimos 365 días?       (binario)
    - fire_cell_count_3y: Nº de veces que ardió la celda en 3 años (1095d) (entero)
    - fire_neighbors_30d: Nº de celdas vecinas a ≤3 km que ardieron en 30d (entero)

Estrategia de implementación (RAM-eficiente):
    1. Cargar solo las filas con fire_occurred=1 (~87K filas, no 136M).
    2. Construir un pivot denso (dias x celdas) de uint8 -> ~136 MB.
    3. Calcular rolling counts con cumsum numpy vectorizado -> O(n_days x n_cells).
    4. Para vecinos: multiplicacion de matriz de adyacencia sparse x rolling count.
    5. Extraer valores por lookup en chunks del dataset completo.

Regla anti-leakage: el periodo de cada lag es [date-N, date-1], nunca incluye
el día actual. Implementado mediante cumsum desplazado (+1 offset).

Uso:
    uv run python -m src.features.fire_history_features
    uv run python -m src.features.fire_history_features --radius-m 3000

Salida:
    data/interim/fire_lag_features.parquet
    Columnas: cell_id, date, fire_cell_lag30d, fire_cell_lag365d,
              fire_cell_count_3y, fire_neighbors_30d
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import typer
from loguru import logger
from scipy.sparse import csr_matrix
from sklearn.neighbors import BallTree

from src.data.config import settings

app = typer.Typer(add_completion=False)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

_DATE_START = "2018-01-01"
_DATE_END = "2024-12-31"
_LAG_WINDOWS: dict[str, int] = {
    "lag30d": 30,
    "lag365d": 365,
    "lag1095d": 1095,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_fire_events(dataset_path: Path) -> pd.DataFrame:
    """Carga únicamente las filas con fire_occurred=1 del dataset principal.

    Args:
        dataset_path: Ruta al archivo Parquet m1_dataset.parquet.

    Returns:
        DataFrame con columnas (cell_id, date) para los eventos de fuego.

    Raises:
        FileNotFoundError: Si el archivo no existe.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset no encontrado: {dataset_path}")

    pf = pq.ParquetFile(str(dataset_path))
    chunks = []
    for batch in pf.iter_batches(
        batch_size=5_000_000, columns=["cell_id", "date", "fire_occurred"]
    ):
        df = batch.to_pandas()
        fires = df[df["fire_occurred"] == 1][["cell_id", "date"]]
        if len(fires) > 0:
            chunks.append(fires)

    fires_df = pd.concat(chunks, ignore_index=True)
    fires_df["date"] = pd.to_datetime(fires_df["date"])
    logger.info(f"Fire events cargados: {len(fires_df):,}")
    return fires_df


def _build_neighbor_lookup(
    grid: gpd.GeoDataFrame,
    radius_m: float,
    id_col: str = "cell_id",
) -> tuple[list[str], csr_matrix]:
    """Construye la matriz de adyacencia sparse para celdas dentro de radio_m.

    Args:
        grid: GeoDataFrame del grid con columnas easting_m, northing_m, {id_col}.
        radius_m: Radio de vecindad en metros.
        id_col: Nombre de la columna de identificador de celda.

    Returns:
        Tupla (cell_ids, adj_matrix) donde adj_matrix es (n_cells x n_cells) sparse.

    Raises:
        ValueError: Si el CRS del grid no es EPSG:32616.
    """
    if grid.crs.to_epsg() != 32616:
        raise ValueError(
            f"CRS del grid debe ser EPSG:32616, encontrado: {grid.crs.to_epsg()}"
        )

    cell_ids = grid[id_col].tolist()
    n_cells = len(cell_ids)
    xy = grid[["easting_m", "northing_m"]].values.astype(np.float64)

    tree = BallTree(xy, metric="euclidean")
    indices_radius = tree.query_radius(xy, r=radius_m)

    rows_sp, cols_sp = [], []
    for i, idx_arr in enumerate(indices_radius):
        for j in idx_arr:
            if j != i:
                rows_sp.append(i)
                cols_sp.append(j)

    adj_matrix = csr_matrix(
        (np.ones(len(rows_sp), dtype=np.float32), (rows_sp, cols_sp)),
        shape=(n_cells, n_cells),
    )
    avg_neighbors = np.mean([len(x) - 1 for x in indices_radius])
    logger.info(
        f"Adyacencia construida: {n_cells:,} celdas, "
        f"{adj_matrix.nnz:,} conexiones, "
        f"promedio {avg_neighbors:.1f} vecinos/celda"
    )
    return cell_ids, adj_matrix


def _rolling_window_count(padded: np.ndarray, window: int) -> np.ndarray:
    """Cuenta fires acumulados en la ventana [d-window, d-1] para cada (d, celda).

    Usa el cumsum desplazado (padded) para cómputo O(n_days·n_cells) sin loops.

    Args:
        padded: Array (n_days+1, n_cells) con cumsum[d] = sum hasta día d-1
                (padded[0] = 0, padded[d] = cumsum de pivot hasta d-1 inclusive).
        window: Tamaño de la ventana en días.

    Returns:
        Array (n_days, n_cells) int32 con el conteo de fires en la ventana.
    """
    n_days = padded.shape[0] - 1
    # result[d] = padded[d] - padded[max(0, d - window)]
    d_indices = np.arange(n_days)
    start_indices = np.maximum(0, d_indices - window)

    result = padded[d_indices] - padded[start_indices]  # broadcasting por filas
    return result.astype(np.int32)


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------


def build_fire_history_features(
    dataset_path: Path | None = None,
    grid_path: Path | None = None,
    output_path: Path | None = None,
    radius_m: float = 3000.0,
    chunk_size: int = 5_000_000,
) -> pd.DataFrame:
    """Construye las features de historial de fuego y las guarda como Parquet.

    Args:
        dataset_path: Ruta al m1_dataset.parquet (default: settings).
        grid_path:    Ruta al peten_grid.gpkg (default: settings).
        output_path:  Destino del Parquet de salida (default: interim/).
        radius_m:     Radio de vecindad en metros para fire_neighbors_30d.
        chunk_size:   Batch size al leer el dataset principal.

    Returns:
        DataFrame con columnas (cell_id, date, fire_cell_lag30d, fire_cell_lag365d,
        fire_cell_count_3y, fire_neighbors_30d).

    Raises:
        FileNotFoundError: Si algún archivo de entrada no existe.
        ValueError:         Si el CRS del grid no es EPSG:32616.
        MemoryError:        Si no hay suficiente RAM para los arrays intermedios.
    """
    dataset_path = dataset_path or (settings.data_processed / "m1_dataset.parquet")
    grid_path = grid_path or (settings.data_interim / "peten_grid.gpkg")
    output_path = output_path or (settings.data_interim / "fire_lag_features.parquet")

    t0 = time.time()

    # ------------------------------------------------------------------
    # 1. Fire events
    # ------------------------------------------------------------------
    logger.info("Paso 1/6: Cargando fire events...")
    fires_df = _load_fire_events(dataset_path)

    # ------------------------------------------------------------------
    # 2. Grid + vecinos
    # ------------------------------------------------------------------
    logger.info("Paso 2/6: Cargando grid y construyendo vecindad...")
    grid = gpd.read_file(str(grid_path))
    if grid.crs.to_epsg() != 32616:
        grid = grid.to_crs(epsg=32616)

    cell_ids, adj_matrix = _build_neighbor_lookup(grid, radius_m=radius_m)
    n_cells = len(cell_ids)
    cell_to_idx: dict[str, int] = {c: i for i, c in enumerate(cell_ids)}

    # ------------------------------------------------------------------
    # 3. Pivot (dias x celdas)
    # ------------------------------------------------------------------
    logger.info("Paso 3/6: Construyendo pivot date x cell...")
    date_range = pd.date_range(_DATE_START, _DATE_END, freq="D")
    n_days = len(date_range)
    date_to_idx: dict[pd.Timestamp, int] = {d: i for i, d in enumerate(date_range)}

    valid_mask = fires_df["cell_id"].isin(cell_to_idx) & fires_df["date"].isin(date_to_idx)
    f = fires_df[valid_mask]
    row_idx = f["date"].map(date_to_idx).values.astype(np.int32)
    col_idx = f["cell_id"].map(cell_to_idx).values.astype(np.int32)

    pivot = np.zeros((n_days, n_cells), dtype=np.uint8)
    np.add.at(pivot, (row_idx, col_idx), 1)
    np.minimum(pivot, 1, out=pivot)
    logger.info(
        f"Pivot: {pivot.shape}, sum={pivot.sum():,}, RAM={pivot.nbytes/1e6:.1f} MB"
    )

    # ------------------------------------------------------------------
    # 4. Cumsum desplazado (anti-leakage: padded[d] = sum hasta d-1)
    # ------------------------------------------------------------------
    logger.info("Paso 4/6: Calculando cumsum y rolling windows...")
    padded = np.zeros((n_days + 1, n_cells), dtype=np.int32)
    padded[1:] = np.cumsum(pivot, axis=0, dtype=np.int32)
    del pivot  # liberar RAM

    lag30 = _rolling_window_count(padded, 30)
    lag365 = _rolling_window_count(padded, 365)
    lag1095 = _rolling_window_count(padded, 1095)
    del padded

    logger.info(
        f"Lags calculados — "
        f"lag30: {lag30.nbytes/1e6:.0f} MB, "
        f"lag365: {lag365.nbytes/1e6:.0f} MB, "
        f"lag1095: {lag1095.nbytes/1e6:.0f} MB"
    )

    # ------------------------------------------------------------------
    # 5. fire_neighbors_30d via multiplicación sparse
    # ------------------------------------------------------------------
    logger.info("Paso 5/6: Calculando fire_neighbors_30d (adj @ lag30)...")
    lag30_transposed = lag30.T.astype(np.float32)  # (n_cells, n_days)
    neighbors_30d = adj_matrix.dot(lag30_transposed).T.astype(np.int16)  # (n_days, n_cells)
    del lag30_transposed
    logger.info(f"neighbors_30d: {neighbors_30d.shape}, RAM={neighbors_30d.nbytes/1e6:.0f} MB")

    # ------------------------------------------------------------------
    # 6. Construcción del DataFrame de salida en chunks
    # ------------------------------------------------------------------
    logger.info("Paso 6/6: Extrayendo features del dataset principal en chunks...")
    pf2 = pq.ParquetFile(str(dataset_path))
    output_chunks: list[pd.DataFrame] = []
    total_processed = 0

    for batch in pf2.iter_batches(batch_size=chunk_size, columns=["cell_id", "date"]):
        df = batch.to_pandas()
        df["date"] = pd.to_datetime(df["date"])

        d_idx = df["date"].map(date_to_idx)
        c_idx = df["cell_id"].map(cell_to_idx)
        valid = d_idx.notna() & c_idx.notna()

        d_safe = d_idx.fillna(0).astype(np.int32).values
        c_safe = c_idx.fillna(0).astype(np.int32).values

        out = pd.DataFrame(
            {
                "cell_id": df["cell_id"],
                "date": df["date"],
                "fire_cell_lag30d": np.where(
                    valid, (lag30[d_safe, c_safe] > 0).astype(np.int8), 0
                ),
                "fire_cell_lag365d": np.where(
                    valid, (lag365[d_safe, c_safe] > 0).astype(np.int8), 0
                ),
                "fire_cell_count_3y": np.where(
                    valid, lag1095[d_safe, c_safe].astype(np.int16), 0
                ),
                "fire_neighbors_30d": np.where(
                    valid, neighbors_30d[d_safe, c_safe].astype(np.int16), 0
                ),
            }
        )
        output_chunks.append(out)
        total_processed += len(df)

    lag_df = pd.concat(output_chunks, ignore_index=True)

    # ------------------------------------------------------------------
    # Guardar
    # ------------------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lag_df.to_parquet(str(output_path), index=False, engine="pyarrow", compression="snappy")
    size_mb = output_path.stat().st_size / 1e6
    elapsed = time.time() - t0

    logger.info(
        f"Guardado: {output_path} — "
        f"{lag_df.shape[0]:,} filas, {lag_df.shape[1]} columnas, {size_mb:.1f} MB"
    )
    logger.info(
        f"Distribución de features:\n"
        f"  fire_cell_lag30d   = 1: {(lag_df['fire_cell_lag30d']==1).sum():,} "
        f"({100*(lag_df['fire_cell_lag30d']==1).mean():.3f}%)\n"
        f"  fire_cell_lag365d  = 1: {(lag_df['fire_cell_lag365d']==1).sum():,} "
        f"({100*(lag_df['fire_cell_lag365d']==1).mean():.3f}%)\n"
        f"  fire_cell_count_3y > 0: {(lag_df['fire_cell_count_3y']>0).sum():,} "
        f"({100*(lag_df['fire_cell_count_3y']>0).mean():.3f}%)\n"
        f"  fire_neighbors_30d > 0: {(lag_df['fire_neighbors_30d']>0).sum():,} "
        f"({100*(lag_df['fire_neighbors_30d']>0).mean():.3f}%)"
    )
    logger.info(f"Tiempo total: {elapsed/60:.1f} minutos")

    return lag_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    radius_m: Annotated[
        float,
        typer.Option("--radius-m", help="Radio de vecindad en metros (default: 3000)"),
    ] = 3000.0,
    chunk_size: Annotated[
        int,
        typer.Option("--chunk-size", help="Batch size para leer el dataset principal"),
    ] = 5_000_000,
) -> None:
    """Construye las fire history features y las guarda en data/interim/."""
    build_fire_history_features(radius_m=radius_m, chunk_size=chunk_size)


if __name__ == "__main__":
    app()

"""Merge NDVI features into the m1_dataset.parquet in a RAM-efficient way.

Strategy: load one year of NDVI at a time (cached), iterate m1_dataset in
batches of 5M rows, merge on (cell_id, date), write to a new parquet file.

Usage:
    uv run python scripts/merge_ndvi_into_dataset.py
"""
from __future__ import annotations

import gc
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

DATA_DIR = Path("data")
NDVI_DIR = DATA_DIR / "interim" / "ndvi"
DATASET_PATH = DATA_DIR / "processed" / "m1_dataset.parquet"
OUTPUT_PATH = DATA_DIR / "processed" / "m1_dataset_new.parquet"
BATCH_SIZE = 5_000_000
NDVI_COLS = ["ndvi", "ndvi_lag7", "ndvi_lag14"]
JOIN_COLS = ["cell_id", "date"]


def load_ndvi_year(year: int) -> pd.DataFrame:
    """Load NDVI features for a given year.

    Args:
        year: Calendar year to load.

    Returns:
        DataFrame with columns [cell_id, date, ndvi, ndvi_lag7, ndvi_lag14].
    """
    path = NDVI_DIR / f"ndvi_features_{year}.parquet"
    if not path.exists():
        logger.warning(f"NDVI parquet not found for year {year}: {path}")
        return pd.DataFrame(columns=JOIN_COLS + NDVI_COLS)
    df = pd.read_parquet(path, columns=JOIN_COLS + NDVI_COLS)
    logger.info(f"Loaded NDVI {year}: {len(df):,} rows")
    return df


def main() -> None:
    """Run the merge pipeline."""
    logger.info(f"Reading dataset: {DATASET_PATH}")
    parquet_file = pq.ParquetFile(str(DATASET_PATH))
    total_rows = parquet_file.metadata.num_rows
    logger.info(f"Total rows to process: {total_rows:,}")

    writer: pq.ParquetWriter | None = None
    rows_written = 0
    current_year: int | None = None
    ndvi_cache: pd.DataFrame = pd.DataFrame()

    for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE):
        chunk = batch.to_pandas()

        # Drop the existing (all-NaN) NDVI columns
        chunk = chunk.drop(columns=NDVI_COLS, errors="ignore")

        # Determine which year(s) are in this chunk
        years_in_chunk = chunk["year"].unique().tolist()

        # For simplicity, process each year group separately within the chunk
        parts: list[pd.DataFrame] = []
        for year in sorted(years_in_chunk):
            year_chunk = chunk[chunk["year"] == year].copy()

            # Load (or reuse cached) NDVI for this year
            if year != current_year:
                logger.info(f"Loading NDVI cache for year {year}...")
                ndvi_cache = load_ndvi_year(year)
                current_year = year
                gc.collect()

            if ndvi_cache.empty:
                for col in NDVI_COLS:
                    year_chunk[col] = float("nan")
            else:
                year_chunk = year_chunk.merge(
                    ndvi_cache, on=JOIN_COLS, how="left"
                )

            parts.append(year_chunk)

        merged = pd.concat(parts, ignore_index=True)

        # Restore original column order
        original_cols = [c for c in batch.schema.names if c not in NDVI_COLS]
        final_cols = original_cols + NDVI_COLS
        merged = merged[final_cols]

        table = pa.Table.from_pandas(merged, preserve_index=False)

        if writer is None:
            writer = pq.ParquetWriter(str(OUTPUT_PATH), table.schema)

        writer.write_table(table)
        rows_written += len(merged)
        logger.info(f"Written {rows_written:,} / {total_rows:,} rows ({rows_written / total_rows * 100:.1f}%)")

    if writer:
        writer.close()

    logger.success(f"Done. Output: {OUTPUT_PATH} ({rows_written:,} rows)")


if __name__ == "__main__":
    main()

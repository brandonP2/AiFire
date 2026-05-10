"""Tests del módulo de entrenamiento M1."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.build_dataset import _assign_split
from src.models.m1_risk.train import (
    _apply_smote,
    _compute_scale_pos_weight,
    _evaluate,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tiny_dataset(tmp_path: Path) -> Path:
    """Dataset M1 mínimo con train/val/test para testing."""
    np.random.seed(42)
    n_total = 600
    cell_ids = [f"r{i:04d}_c0000" for i in range(10)] * 60

    dates_train = [date(2020, 3, d) for d in range(1, 31)] * 2
    dates_val = [date(2023, 3, d) for d in range(1, 31)]
    dates_test = [date(2024, 3, d) for d in range(1, 31)]
    all_dates = (dates_train * 5 + dates_val * 2 + dates_test * 2)[:n_total]

    df = pd.DataFrame(
        {
            "cell_id": cell_ids[:n_total],
            "date": all_dates,
            "fire_occurred": np.random.choice([True, False], size=n_total, p=[0.02, 0.98]),
            "T2M": np.random.uniform(20, 38, n_total),
            "RH2M": np.random.uniform(20, 80, n_total),
            "WS10M": np.random.uniform(0, 30, n_total),
            "PRECTOTCORR": np.random.exponential(1, n_total),
            "fwi": np.random.uniform(0, 60, n_total),
            "ffmc_val": np.random.uniform(70, 100, n_total),
            "dmc_val": np.random.uniform(0, 100, n_total),
            "dc_val": np.random.uniform(0, 500, n_total),
            "isi_val": np.random.uniform(0, 20, n_total),
            "bui_val": np.random.uniform(0, 100, n_total),
            "prec_acc7d": np.random.exponential(5, n_total),
            "prec_acc14d": np.random.exponential(10, n_total),
            "ndvi": np.random.uniform(0.1, 0.8, n_total),
            "ndvi_lag7": np.random.uniform(0.1, 0.8, n_total),
            "ndvi_lag14": np.random.uniform(0.1, 0.8, n_total),
            "elevation_m": np.random.uniform(100, 400, n_total),
            "slope_deg": np.random.uniform(0, 30, n_total),
            "aspect_deg": np.random.uniform(0, 360, n_total),
            "dist_roads_km": np.random.uniform(0, 50, n_total),
            "dist_settlements_km": np.random.uniform(0, 80, n_total),
            "is_protected_area": np.random.choice([True, False], n_total),
            "month": [d.month for d in all_dates],
            "day_of_year": [d.timetuple().tm_yday for d in all_dates],
            "year": [d.year for d in all_dates],
        }
    )

    # Asignar split según año
    df["split"] = "train"
    df.loc[df["year"] == 2023, "split"] = "val"
    df.loc[df["year"] == 2024, "split"] = "test"

    output = tmp_path / "m1_dataset.parquet"
    df.to_parquet(output, index=False)
    return output


# ---------------------------------------------------------------------------
# Tests de helpers
# ---------------------------------------------------------------------------


def test_assign_split_train() -> None:
    dates = pd.Series([date(2020, 3, 1), date(2022, 12, 31)])
    splits = _assign_split(dates)
    assert list(splits) == ["train", "train"]


def test_assign_split_val() -> None:
    dates = pd.Series([date(2023, 1, 1), date(2023, 12, 31)])
    splits = _assign_split(dates)
    assert list(splits) == ["val", "val"]


def test_assign_split_test() -> None:
    dates = pd.Series([date(2024, 6, 15)])
    splits = _assign_split(dates)
    assert splits.iloc[0] == "test"


def test_compute_scale_pos_weight() -> None:
    y = np.array([0] * 98 + [1] * 2)
    spw = _compute_scale_pos_weight(y)
    assert abs(spw - 49.0) < 0.1


def test_compute_scale_pos_weight_no_positives() -> None:
    y = np.zeros(100, dtype=int)
    with pytest.raises(ValueError, match="No hay muestras positivas"):
        _compute_scale_pos_weight(y)


# ---------------------------------------------------------------------------
# Tests de SMOTE
# ---------------------------------------------------------------------------


def test_apply_smote_increases_minority() -> None:
    np.random.seed(42)
    X = np.random.rand(1000, 5).astype(np.float32)
    y = np.array([0] * 980 + [1] * 20)

    _x_res, y_res = _apply_smote(X, y, random_state=42, sampling_strategy=0.1)

    original_rate = y.mean()
    new_rate = y_res.mean()
    assert new_rate > original_rate


def test_apply_smote_output_shape() -> None:
    np.random.seed(0)
    X = np.random.rand(500, 10).astype(np.float32)
    y = np.array([0] * 490 + [1] * 10)

    X_res, y_res = _apply_smote(X, y, random_state=0, sampling_strategy=0.05)

    assert X_res.shape[1] == X.shape[1]
    assert len(X_res) == len(y_res)


# ---------------------------------------------------------------------------
# Tests de evaluación
# ---------------------------------------------------------------------------


def test_evaluate_returns_expected_metrics() -> None:
    """_evaluate debe retornar las métricas clave."""
    from sklearn.linear_model import LogisticRegression

    np.random.seed(42)
    X = np.random.rand(200, 3)
    y = np.random.choice([0, 1], 200, p=[0.95, 0.05])

    lr = LogisticRegression(class_weight="balanced", random_state=42, max_iter=200)
    lr.fit(X, y)

    metrics = _evaluate(lr, X, y, "test", threshold=0.5)

    assert set(metrics.keys()) == {"auc_roc", "auc_pr", "precision", "recall", "f1"}
    for v in metrics.values():
        assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# Tests de integración (end-to-end con modelo liviano)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_train_m1_lightgbm_smoke(tmp_path: Path, tiny_dataset: Path) -> None:
    """Smoke test: train_m1 con LightGBM debe completar sin error."""
    from src.models.m1_risk.train import train_m1

    output_dir = tmp_path / "model_output"
    metrics = train_m1(
        dataset_path=tiny_dataset,
        output_dir=output_dir,
        model_type="lightgbm",
        n_estimators=50,
        use_smote=False,
    )

    assert "val" in metrics
    assert "test" in metrics
    assert metrics["val"]["auc_roc"] > 0.0
    assert (output_dir / "m1_lightgbm_calibrated.joblib").exists()
    assert (output_dir / "metrics.yaml").exists()

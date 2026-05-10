from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.download_firms import _fetch_chunk, _bbox_str, download_firms, SOURCES

MOCK_CSV_VIIRS = (
    "latitude,longitude,bright_ti4,bright_ti5,scan,track,acq_date,acq_time,"
    "satellite,instrument,confidence,version,bright_ti5,frp,daynight\n"
    "16.5,-90.3,340.2,310.1,0.4,0.4,2024-03-15,1230,N,VIIRS,nominal,2.0NRT,310.1,5.2,D\n"
    "17.0,-90.8,345.0,312.0,0.4,0.4,2024-03-15,1430,N,VIIRS,high,2.0NRT,312.0,8.1,D\n"
)

MOCK_CSV_MODIS = (
    "latitude,longitude,brightness,scan,track,acq_date,acq_time,"
    "satellite,instrument,confidence,version,bright_t31,frp,daynight\n"
    "16.8,-90.5,332.1,1.0,1.0,2024-03-15,1545,Terra,MODIS,50,6.1,305.0,12.3,D\n"
)


def _mock_response(text: str, status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.text = text
    return resp


# ---------------------------------------------------------------------------
# _fetch_chunk
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("source,csv_text,expected_col", [
    ("VIIRS_SNPP_NRT", MOCK_CSV_VIIRS, "bright_ti4"),
    ("MODIS_NRT", MOCK_CSV_MODIS, "bright_ti4"),  # brightness → bright_ti4
])
def test_fetch_chunk_returns_dataframe(source: str, csv_text: str, expected_col: str) -> None:
    with patch("httpx.get", return_value=_mock_response(csv_text)):
        df = _fetch_chunk("fake_key", source, "-91.45,15.88,-89.14,17.82", date(2024, 3, 15), 1)

    assert df is not None
    assert not df.empty
    assert expected_col in df.columns
    assert "source" in df.columns
    assert (df["source"] == source).all()


def test_fetch_chunk_empty_response_returns_empty_df() -> None:
    with patch("httpx.get", return_value=_mock_response("latitude,longitude,acq_date\n")):
        df = _fetch_chunk("fake_key", "VIIRS_SNPP_NRT", "bbox", date(2024, 3, 15), 1)
    assert df is not None
    assert df.empty


def test_fetch_chunk_http_error_returns_none() -> None:
    with patch("httpx.get", return_value=_mock_response("Forbidden", status=403)):
        df = _fetch_chunk("bad_key", "VIIRS_SNPP_NRT", "bbox", date(2024, 3, 15), 1)
    assert df is None


def test_fetch_chunk_network_error_retries() -> None:
    import httpx as httpx_module
    with patch("httpx.get", side_effect=httpx_module.RequestError("timeout")):
        with patch("time.sleep"):
            df = _fetch_chunk("key", "VIIRS_SNPP_NRT", "bbox", date(2024, 3, 15), 1, max_retries=2)
    assert df is None


# ---------------------------------------------------------------------------
# _bbox_str
# ---------------------------------------------------------------------------

def test_bbox_str_format() -> None:
    result = _bbox_str((-91.45, 15.88, -89.14, 17.82))
    assert result == "-91.45,15.88,-89.14,17.82"


# ---------------------------------------------------------------------------
# download_firms
# ---------------------------------------------------------------------------

def test_download_firms_invalid_source_raises() -> None:
    with pytest.raises(ValueError, match="Fuente inválida"):
        download_firms(
            start=date(2024, 1, 1),
            end=date(2024, 1, 2),
            source="INVALID_SOURCE",
            map_key="key",
            bbox=(-91.45, 15.88, -89.14, 17.82),
            output_dir=Path("/tmp"),
        )


def test_download_firms_skips_existing_file(tmp_path: Path) -> None:
    existing = tmp_path / "firms_VIIRS_SNPP_NRT_2024-01_to_2024-01.csv"
    existing.write_text("dummy")

    with patch("httpx.get") as mock_get:
        result = download_firms(
            start=date(2024, 1, 1),
            end=date(2024, 1, 5),
            source="VIIRS_SNPP_NRT",
            map_key="key",
            bbox=(-91.45, 15.88, -89.14, 17.82),
            output_dir=tmp_path,
        )
    mock_get.assert_not_called()
    assert result == existing


def test_download_firms_writes_csv(tmp_path: Path) -> None:
    with patch("httpx.get", return_value=_mock_response(MOCK_CSV_VIIRS)):
        with patch("time.sleep"):
            result = download_firms(
                start=date(2024, 3, 15),
                end=date(2024, 3, 16),
                source="VIIRS_SNPP_NRT",
                map_key="fake_key",
                bbox=(-91.45, 15.88, -89.14, 17.82),
                output_dir=tmp_path,
                chunk_days=2,
            )

    assert result is not None
    assert result.exists()
    df = pd.read_csv(result)
    assert "latitude" in df.columns
    assert "source" in df.columns


def test_sources_constant_not_empty() -> None:
    assert len(SOURCES) >= 2
    assert "VIIRS_SNPP_NRT" in SOURCES
    assert "MODIS_NRT" in SOURCES

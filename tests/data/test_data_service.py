from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from config.settings import CacheSettings, Settings
from data.cache.cache_manager import CacheManager
from data.data_service import DataService


@pytest.fixture
def cache(tmp_path: Path) -> CacheManager:
    return CacheManager(CacheSettings(cache_dir=tmp_path / "cache", max_age_hours=24))


@pytest.fixture
def mock_loader(sample_price_df: pd.DataFrame) -> MagicMock:
    loader = MagicMock()
    loader.fetch.return_value = sample_price_df
    loader.fetch_batch.return_value = {"AAPL": sample_price_df}
    return loader


@pytest.fixture
def service(mock_loader: MagicMock, cache: CacheManager) -> DataService:
    return DataService(loader=mock_loader, cache=cache, settings=Settings.default())


# ------------------------------------------------------------------
# get (single ticker)
# ------------------------------------------------------------------


def test_get_fetches_and_caches(service: DataService, mock_loader: MagicMock) -> None:
    df = service.get("AAPL", date(2024, 1, 2), date(2024, 1, 8))
    assert not df.empty
    mock_loader.fetch.assert_called_once()


def test_get_uses_cache_on_second_call(
    service: DataService, mock_loader: MagicMock
) -> None:
    service.get("AAPL", date(2024, 1, 2), date(2024, 1, 8))
    service.get("AAPL", date(2024, 1, 2), date(2024, 1, 8))
    # Second call should be served from cache
    mock_loader.fetch.assert_called_once()


def test_get_force_refresh_bypasses_cache(
    service: DataService, mock_loader: MagicMock
) -> None:
    service.get("AAPL", date(2024, 1, 2), date(2024, 1, 8))
    service.get("AAPL", date(2024, 1, 2), date(2024, 1, 8), force_refresh=True)
    assert mock_loader.fetch.call_count == 2


def test_get_uses_default_date_range_when_none(
    service: DataService, mock_loader: MagicMock
) -> None:
    df = service.get("AAPL")
    assert not df.empty
    mock_loader.fetch.assert_called_once()


# ------------------------------------------------------------------
# get_batch (multiple tickers)
# ------------------------------------------------------------------


def test_get_batch_returns_dict(service: DataService, mock_loader: MagicMock) -> None:
    result = service.get_batch(["AAPL"], date(2024, 1, 2), date(2024, 1, 8))
    assert "AAPL" in result
    mock_loader.fetch_batch.assert_called_once()


def test_get_batch_cached_tickers_skip_loader(
    service: DataService, mock_loader: MagicMock
) -> None:
    # First call: fetch and cache AAPL
    service.get_batch(["AAPL"], date(2024, 1, 2), date(2024, 1, 8))
    # Second call: AAPL is fresh and covers range, loader should NOT be called again
    service.get_batch(["AAPL"], date(2024, 1, 2), date(2024, 1, 8))
    mock_loader.fetch_batch.assert_called_once()


def test_get_batch_empty_list_returns_empty(service: DataService) -> None:
    result = service.get_batch([])
    assert result == {}

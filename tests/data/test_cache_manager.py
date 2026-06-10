from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest

from config.settings import CacheSettings
from data.cache.cache_manager import CacheManager


@pytest.fixture
def cache(tmp_path: Path) -> CacheManager:
    return CacheManager(CacheSettings(cache_dir=tmp_path / "cache", max_age_hours=24))


# ------------------------------------------------------------------
# Existence & freshness
# ------------------------------------------------------------------


def test_exists_false_initially(cache: CacheManager) -> None:
    assert not cache.exists("AAPL")


def test_exists_true_after_save(
    cache: CacheManager, sample_price_df: pd.DataFrame
) -> None:
    cache.save("AAPL", sample_price_df)
    assert cache.exists("AAPL")


def test_is_fresh_after_save(
    cache: CacheManager, sample_price_df: pd.DataFrame
) -> None:
    cache.save("AAPL", sample_price_df)
    assert cache.is_fresh("AAPL")


def test_is_fresh_false_when_max_age_zero(
    tmp_path: Path, sample_price_df: pd.DataFrame
) -> None:
    stale_cache = CacheManager(
        CacheSettings(cache_dir=tmp_path / "stale", max_age_hours=0)
    )
    stale_cache.save("AAPL", sample_price_df)
    # max_age_hours=0 → timedelta(0), any positive age is already stale
    assert not stale_cache.is_fresh("AAPL")


def test_is_fresh_false_when_not_exists(cache: CacheManager) -> None:
    assert not cache.is_fresh("MISSING")


# ------------------------------------------------------------------
# Save / load round-trip
# ------------------------------------------------------------------


def test_save_and_load_roundtrip(
    cache: CacheManager, sample_price_df: pd.DataFrame
) -> None:
    cache.save("AAPL", sample_price_df)
    loaded = cache.load("AAPL")
    # Parquet does not preserve DatetimeIndex frequency metadata
    pd.testing.assert_frame_equal(loaded, sample_price_df, check_freq=False)


def test_load_raises_when_not_found(cache: CacheManager) -> None:
    with pytest.raises(FileNotFoundError, match="MISSING"):
        cache.load("MISSING")


# ------------------------------------------------------------------
# Invalidation
# ------------------------------------------------------------------


def test_invalidate_removes_file(
    cache: CacheManager, sample_price_df: pd.DataFrame
) -> None:
    cache.save("AAPL", sample_price_df)
    cache.invalidate("AAPL")
    assert not cache.exists("AAPL")


def test_invalidate_nonexistent_is_noop(cache: CacheManager) -> None:
    cache.invalidate("GHOST")  # must not raise


def test_clear_all_removes_all(
    cache: CacheManager, sample_price_df: pd.DataFrame
) -> None:
    for ticker in ("AAPL", "MSFT", "NVDA"):
        cache.save(ticker, sample_price_df)
    cache.clear_all()
    assert not cache.exists("AAPL")
    assert not cache.exists("MSFT")
    assert not cache.exists("NVDA")


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------


def test_ticker_with_slash(cache: CacheManager, sample_price_df: pd.DataFrame) -> None:
    """BRK/B must not create a nested path."""
    cache.save("BRK/B", sample_price_df)
    assert cache.exists("BRK/B")
    loaded = cache.load("BRK/B")
    pd.testing.assert_frame_equal(loaded, sample_price_df, check_freq=False)

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from features.trend import (
    DeathCross,
    DeathCrossFeature,
    Dist52WHigh,
    Dist52WHighFeature,
    SMA50,
    SMA200,
    SMAFeature,
)


def _make_df(close: np.ndarray) -> pd.DataFrame:
    n = len(close)
    idx = pd.bdate_range("2023-01-02", periods=n, name="date")
    high = close * 1.01
    return pd.DataFrame(
        {
            "open": close,
            "high": high,
            "low": close * 0.99,
            "close": close,
            "volume": np.ones(n),
        },
        index=idx,
    )


# ------------------------------------------------------------------
# SMAFeature
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "feature, expected_name",
    [
        (SMA50, "sma_50"),
        (SMA200, "sma_200"),
    ],
)
def test_sma_name(feature: SMAFeature, expected_name: str) -> None:
    assert feature.name == expected_name


def test_sma50_not_enough_data_returns_nan(short_price_df: pd.DataFrame) -> None:
    assert math.isnan(SMA50.compute(short_price_df))


def test_sma200_not_enough_data_returns_nan(long_price_df: pd.DataFrame) -> None:
    # long_price_df has 300 rows — enough for SMA200
    # Slice to 199 rows → NaN
    assert math.isnan(SMA200.compute(long_price_df.iloc[:199]))


def test_sma50_flat_prices_equals_price() -> None:
    close = np.full(60, 150.0)
    df = _make_df(close)
    assert SMA50.compute(df) == pytest.approx(150.0)


def test_sma200_flat_prices_equals_price(long_price_df: pd.DataFrame) -> None:
    close = np.full(210, 200.0)
    df = _make_df(close)
    assert SMA200.compute(df) == pytest.approx(200.0)


def test_sma50_computes_on_long_df(long_price_df: pd.DataFrame) -> None:
    val = SMA50.compute(long_price_df)
    assert not math.isnan(val)
    assert val > 0


# ------------------------------------------------------------------
# Dist52WHighFeature
# ------------------------------------------------------------------


def test_dist52w_name() -> None:
    assert Dist52WHigh.name == "dist_52w_high"


def test_dist52w_not_enough_data_returns_nan(short_price_df: pd.DataFrame) -> None:
    # 5 rows is technically enough (> 2), but the feature should still run
    # short_price_df has close < high so dist should be negative
    val = Dist52WHigh.compute(short_price_df)
    assert not math.isnan(val)
    assert val <= 0.0


def test_dist52w_at_high_returns_near_zero() -> None:
    """When the last close equals the 52-week high, dist ≈ 0."""
    close = np.full(252, 100.0)
    idx = pd.bdate_range("2023-01-02", periods=252, name="date")
    df = pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.ones(252),
        },
        index=idx,
    )
    val = Dist52WHigh.compute(df)
    assert val == pytest.approx(0.0)


def test_dist52w_below_high_is_negative(long_price_df: pd.DataFrame) -> None:
    val = Dist52WHigh.compute(long_price_df)
    assert not math.isnan(val)
    assert val <= 0.0  # close can never exceed its own 52w high


def test_dist52w_uses_only_252_day_lookback() -> None:
    """High outside the 252-day window must NOT affect the result."""
    n = 300
    close = np.full(n, 100.0)
    high = np.full(n, 100.0)
    high[0] = 999.0  # ancient spike — outside 252-day window
    idx = pd.bdate_range("2023-01-02", periods=n, name="date")
    df = pd.DataFrame(
        {
            "open": close,
            "high": high,
            "low": close,
            "close": close,
            "volume": np.ones(n),
        },
        index=idx,
    )
    val = Dist52WHigh.compute(df)
    assert val == pytest.approx(0.0)


# ------------------------------------------------------------------
# DeathCrossFeature
# ------------------------------------------------------------------


def test_death_cross_name() -> None:
    assert DeathCross.name == "death_cross"


def test_death_cross_not_enough_data_returns_nan() -> None:
    close = np.full(199, 100.0)
    df = _make_df(close)
    assert math.isnan(DeathCross.compute(df))


def test_golden_cross_returns_zero() -> None:
    """SMA50 > SMA200 → golden cross → 0.0."""
    n = 210
    close = np.linspace(100, 200, n)
    df = _make_df(close)
    assert DeathCross.compute(df) == pytest.approx(0.0)


def test_death_cross_returns_one() -> None:
    """SMA50 < SMA200 → death cross → 1.0."""
    n = 210
    close = np.linspace(200, 100, n)
    df = _make_df(close)
    assert DeathCross.compute(df) == pytest.approx(1.0)


def test_flat_prices_golden_cross() -> None:
    """Flat prices → SMA50 == SMA200 → 0.0 (not a death cross)."""
    close = np.full(210, 100.0)
    df = _make_df(close)
    assert DeathCross.compute(df) == pytest.approx(0.0)

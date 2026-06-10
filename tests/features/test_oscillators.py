from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from features.oscillators import RSI14, RSIFeature


def _make_df(close: np.ndarray) -> pd.DataFrame:
    n = len(close)
    idx = pd.bdate_range("2024-01-02", periods=n, name="date")
    return pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.ones(n),
        },
        index=idx,
    )


# ------------------------------------------------------------------
# Name
# ------------------------------------------------------------------


def test_rsi14_name() -> None:
    assert RSI14.name == "rsi_14"


def test_rsi_custom_period_name() -> None:
    assert RSIFeature(period=9).name == "rsi_9"


# ------------------------------------------------------------------
# Not enough data
# ------------------------------------------------------------------


def test_rsi_not_enough_data_returns_nan(short_price_df: pd.DataFrame) -> None:
    assert math.isnan(RSI14.compute(short_price_df))


def test_rsi_exactly_at_threshold_returns_nan() -> None:
    # period=14 requires at least 15 rows; give exactly 14
    df = _make_df(np.arange(1.0, 15.0))
    assert math.isnan(RSI14.compute(df))


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------


def test_rsi_all_gains_returns_100() -> None:
    close = np.linspace(100, 130, 30)
    df = _make_df(close)
    assert RSI14.compute(df) == pytest.approx(100.0)


def test_rsi_all_losses_returns_zero() -> None:
    close = np.linspace(130, 100, 30)
    df = _make_df(close)
    assert RSI14.compute(df) == pytest.approx(0.0)


# ------------------------------------------------------------------
# Range and determinism
# ------------------------------------------------------------------


def test_rsi_in_valid_range(long_price_df: pd.DataFrame) -> None:
    val = RSI14.compute(long_price_df)
    assert not math.isnan(val)
    assert 0.0 <= val <= 100.0


def test_rsi_deterministic(long_price_df: pd.DataFrame) -> None:
    assert RSI14.compute(long_price_df) == RSI14.compute(long_price_df)


# ------------------------------------------------------------------
# Backtest slice
# ------------------------------------------------------------------


def test_rsi_changes_on_slice(long_price_df: pd.DataFrame) -> None:
    full_val = RSI14.compute(long_price_df)
    sliced_val = RSI14.compute(long_price_df.iloc[:-30])
    assert full_val != pytest.approx(sliced_val)

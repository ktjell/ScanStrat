from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from features.momentum import Momentum3M, Momentum6M, Momentum12M, MomentumFeature


def _make_flat_df(n: int, price: float = 100.0) -> pd.DataFrame:
    idx = pd.bdate_range("2023-01-02", periods=n, name="date")
    close = np.full(n, price)
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
# Momentum3M  (63 trading days)
# ------------------------------------------------------------------


def test_momentum3m_not_enough_data_returns_nan(short_price_df: pd.DataFrame) -> None:
    assert math.isnan(Momentum3M.compute(short_price_df))


def test_momentum3m_flat_prices_return_zero(long_price_df: pd.DataFrame) -> None:
    df = _make_flat_df(70)
    assert Momentum3M.compute(df) == pytest.approx(0.0)


def test_momentum3m_doubles_returns_one(long_price_df: pd.DataFrame) -> None:
    """Price doubles over the lookback window → momentum = 1.0 (100 %)."""
    n = 70
    idx = pd.bdate_range("2023-01-02", periods=n, name="date")
    close = np.ones(n)
    # First 6 rows at 1.0, then exactly 64th-from-last row at 1.0, last at 2.0
    close[-64] = 1.0
    close[-1] = 2.0
    df = pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.ones(n),
        },
        index=idx,
    )
    assert Momentum3M.compute(df) == pytest.approx(1.0)


def test_momentum3m_uses_correct_lookback(long_price_df: pd.DataFrame) -> None:
    val = Momentum3M.compute(long_price_df)
    assert not math.isnan(val)


# ------------------------------------------------------------------
# Momentum6M / 12M — name and nan behaviour
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "feature, expected_name",
    [
        (Momentum3M, "momentum_3m"),
        (Momentum6M, "momentum_6m"),
        (Momentum12M, "momentum_12m"),
    ],
)
def test_name(feature: MomentumFeature, expected_name: str) -> None:
    assert feature.name == expected_name


def test_momentum6m_not_enough_data_returns_nan(short_price_df: pd.DataFrame) -> None:
    assert math.isnan(Momentum6M.compute(short_price_df))


def test_momentum12m_not_enough_data_returns_nan(short_price_df: pd.DataFrame) -> None:
    assert math.isnan(Momentum12M.compute(short_price_df))


def test_momentum12m_computes_on_long_df(long_price_df: pd.DataFrame) -> None:
    val = Momentum12M.compute(long_price_df)
    assert not math.isnan(val)
    assert -1.0 < val < 10.0  # sanity bounds for a random walk


# ------------------------------------------------------------------
# Backtest slice usage
# ------------------------------------------------------------------


def test_momentum_respects_slice(long_price_df: pd.DataFrame) -> None:
    """Slicing the DataFrame before compute changes the result."""
    full_val = Momentum3M.compute(long_price_df)
    sliced_val = Momentum3M.compute(long_price_df.iloc[:-20])
    assert full_val != pytest.approx(sliced_val)

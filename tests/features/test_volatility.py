from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from features.volatility import Volatility30D, VolatilityFeature


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


def test_volatility30d_name() -> None:
    assert Volatility30D.name == "volatility_30d"


def test_custom_window_name() -> None:
    assert VolatilityFeature(window=60).name == "volatility_60d"


# ------------------------------------------------------------------
# Not enough data
# ------------------------------------------------------------------


def test_not_enough_data_returns_nan(short_price_df: pd.DataFrame) -> None:
    assert math.isnan(Volatility30D.compute(short_price_df))


def test_exactly_at_threshold_returns_nan() -> None:
    # window=30 requires at least 31 rows
    df = _make_df(np.linspace(100, 110, 30))
    assert math.isnan(Volatility30D.compute(df))


# ------------------------------------------------------------------
# Flat prices → zero volatility
# ------------------------------------------------------------------


def test_flat_prices_returns_zero() -> None:
    close = np.full(40, 100.0)
    df = _make_df(close)
    assert Volatility30D.compute(df) == pytest.approx(0.0)


# ------------------------------------------------------------------
# Range and annualization
# ------------------------------------------------------------------


def test_volatility_is_positive_for_random_walk(long_price_df: pd.DataFrame) -> None:
    val = Volatility30D.compute(long_price_df)
    assert not math.isnan(val)
    assert val > 0.0


def test_volatility_is_annualized(long_price_df: pd.DataFrame) -> None:
    """Annualized vol for a ~1.5 % daily std process should be in a plausible range."""
    val = Volatility30D.compute(long_price_df)
    # 1.5 % daily * sqrt(252) ≈ 0.24; reasonable range 0.10 – 0.60
    assert 0.05 < val < 1.0


def test_volatility_uses_only_window_rows() -> None:
    """Rows outside the window must not affect the result."""
    rng = np.random.default_rng(0)
    n = 100
    close_base = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    df_base = _make_df(close_base)
    val_base = Volatility30D.compute(df_base)

    # Prepend 50 rows with extreme volatility
    close_spike = np.array([1.0, 1000.0] * 25)
    close_combined = np.concatenate([close_spike, close_base])
    df_combined = _make_df(close_combined)
    val_combined = Volatility30D.compute(df_combined)

    assert val_base == pytest.approx(val_combined, rel=1e-6)

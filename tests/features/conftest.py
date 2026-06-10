from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def long_price_df() -> pd.DataFrame:
    """
    300 business-day OHLCV DataFrame — enough for SMA200, 12m momentum, etc.
    Uses a seeded random walk so tests are deterministic.
    """
    rng = np.random.default_rng(42)
    n = 300
    idx = pd.bdate_range("2023-01-02", periods=n, name="date")
    log_returns = rng.normal(0.0003, 0.015, n)
    close = 100.0 * np.exp(np.cumsum(log_returns))
    high = close * (1 + rng.uniform(0.002, 0.015, n))
    low = close * (1 - rng.uniform(0.002, 0.015, n))
    return pd.DataFrame(
        {
            "open": close * (1 + rng.uniform(-0.005, 0.005, n)),
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full(n, 1_000_000.0),
        },
        index=idx,
    )


@pytest.fixture
def short_price_df() -> pd.DataFrame:
    """5-row DataFrame — too short for most features (returns NaN)."""
    idx = pd.bdate_range("2024-01-02", periods=5, name="date")
    close = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(5, 1_000_000.0),
        },
        index=idx,
    )

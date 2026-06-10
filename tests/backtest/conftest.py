from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers shared across test files in this package
# ---------------------------------------------------------------------------


def make_price_df(start: str, periods: int, price: float = 100.0) -> pd.DataFrame:
    """Flat-price OHLCV DataFrame with a business-day index."""
    idx = pd.bdate_range(start, periods=periods, name="date")
    return pd.DataFrame(
        {
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": 1_000_000,
        },
        index=idx,
    )


def make_rising_df(
    start: str, periods: int, start_price: float = 100.0, end_price: float = 200.0
) -> pd.DataFrame:
    """Price that rises linearly from start_price to end_price."""
    idx = pd.bdate_range(start, periods=periods, name="date")
    prices = np.linspace(start_price, end_price, periods)
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": 1_000_000,
        },
        index=idx,
    )

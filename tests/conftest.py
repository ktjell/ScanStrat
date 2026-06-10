from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def sample_price_df() -> pd.DataFrame:
    """
    Minimal canonical OHLCV DataFrame that conforms to the price schema.
    Reused across all test modules.
    """
    idx = pd.date_range("2024-01-02", periods=5, freq="B", name="date")
    return pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [105.0, 106.0, 107.0, 108.0, 109.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [102.0, 103.0, 104.0, 105.0, 106.0],
            "volume": [1_000_000.0] * 5,
        },
        index=idx,
    )

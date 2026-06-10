from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def feature_df() -> pd.DataFrame:
    """
    5-ticker feature DataFrame as FeatureEngine would produce.
    Values are chosen so ranking order is unambiguous.
    """
    data = {
        "momentum_12m": [0.40, 0.30, 0.20, 0.10, -0.05],
        "momentum_6m": [0.20, 0.15, 0.10, 0.05, 0.00],
        "momentum_3m": [0.10, 0.08, 0.06, 0.03, 0.00],
        "dist_52w_high": [-0.01, -0.03, -0.07, -0.12, -0.20],
        "rsi_14": [60.0, 55.0, 50.0, 45.0, 35.0],
        "volatility_30d": [0.15, 0.18, 0.22, 0.28, 0.35],
    }
    tickers = ["NVDA", "MSFT", "AAPL", "META", "TSLA"]
    return pd.DataFrame(data, index=pd.Index(tickers, name="ticker"))

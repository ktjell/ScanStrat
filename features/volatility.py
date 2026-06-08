from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VolatilityFeature:
    """
    Annualized historical volatility based on log returns.

    Formula: std(log(close[t] / close[t-1]), window) * sqrt(252)

    A window of 30 trading days (~6 weeks) is the default.
    """

    window: int = 30

    @property
    def name(self) -> str:
        return f"volatility_{self.window}d"

    def compute(self, df: pd.DataFrame) -> float:
        close = df["close"].dropna()
        if len(close) < self.window + 1:
            return float("nan")
        log_returns = np.log(close / close.shift(1)).dropna()
        vol = float(log_returns.iloc[-self.window :].std() * np.sqrt(252))
        return vol


Volatility30D = VolatilityFeature(window=30)

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class MomentumFeature:
    """
    Price return over a fixed lookback window (in trading days).

    Formula: close[-1] / close[-(trading_days + 1)] - 1

    Skip-last-month adjustment is intentionally omitted; apply it at the
    ranking layer if desired.
    """

    trading_days: int

    @property
    def name(self) -> str:
        months = round(self.trading_days / 21)
        return f"momentum_{months}m"

    def compute(self, df: pd.DataFrame) -> float:
        close = df["close"].dropna()
        required = self.trading_days + 1
        if len(close) < required:
            return float("nan")
        return float(close.iloc[-1] / close.iloc[-required] - 1)


Momentum3M = MomentumFeature(trading_days=63)
Momentum6M = MomentumFeature(trading_days=126)
Momentum12M = MomentumFeature(trading_days=252)

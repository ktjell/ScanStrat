from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class RSIFeature:
    """
    Relative Strength Index using Wilder's exponential smoothing.

    Wilder's smoothing ≡ pandas EWM with alpha=1/period, adjust=False.

    Returns NaN when there is insufficient data (< period + 1 rows).
    Returns 100.0 when average loss is zero (all up-days).
    Returns 0.0  when average gain is zero (all down-days).
    """

    period: int = 14

    @property
    def name(self) -> str:
        return f"rsi_{self.period}"

    def compute(self, df: pd.DataFrame) -> float:
        close = df["close"].dropna()
        if len(close) < self.period + 1:
            return float("nan")

        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

        alpha = 1.0 / self.period
        avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=self.period).mean()
        avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=self.period).mean()

        last_gain = avg_gain.iloc[-1]
        last_loss = avg_loss.iloc[-1]

        if pd.isna(last_gain) or pd.isna(last_loss):
            return float("nan")
        if last_loss == 0.0:
            return 100.0
        if last_gain == 0.0:
            return 0.0

        rs = last_gain / last_loss
        return float(100.0 - 100.0 / (1.0 + rs))


RSI14 = RSIFeature(period=14)

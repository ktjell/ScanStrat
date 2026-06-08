from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SMAFeature:
    """Simple Moving Average of the close price."""

    period: int

    @property
    def name(self) -> str:
        return f"sma_{self.period}"

    def compute(self, df: pd.DataFrame) -> float:
        close = df["close"].dropna()
        if len(close) < self.period:
            return float("nan")
        return float(close.rolling(self.period).mean().iloc[-1])


class Dist52WHighFeature:
    """
    Distance of the latest close from the 52-week high.

    Formula: close[-1] / max(high[-252:]) - 1

    A value of -0.04 means the stock is 4 % below its 52-week high.
    """

    name: str = "dist_52w_high"

    def compute(self, df: pd.DataFrame) -> float:
        close = df["close"].dropna()
        high = df["high"].dropna()
        if len(close) < 2 or len(high) < 2:
            return float("nan")
        lookback = min(252, len(high))
        high_52w = float(high.iloc[-lookback:].max())
        if high_52w == 0:
            return float("nan")
        return float(close.iloc[-1] / high_52w - 1)


SMA50 = SMAFeature(period=50)
SMA200 = SMAFeature(period=200)
Dist52WHigh = Dist52WHighFeature()


class DeathCrossFeature:
    """
    Death cross / golden cross indicator.

    Returns
    -------
    1.0  — death cross  (SMA50 < SMA200, bearish)
    0.0  — golden cross (SMA50 >= SMA200, bullish)
    nan  — insufficient data (< 200 rows)

    In the Normalizer this feature uses ascending=False so that
    0.0 (golden cross) maps to a high percentile score.
    """

    name: str = "death_cross"

    def compute(self, df: pd.DataFrame) -> float:
        close = df["close"].dropna()
        if len(close) < 200:
            return float("nan")
        sma50 = float(close.rolling(50).mean().iloc[-1])
        sma200 = float(close.rolling(200).mean().iloc[-1])
        return 1.0 if sma50 < sma200 else 0.0


DeathCross = DeathCrossFeature()

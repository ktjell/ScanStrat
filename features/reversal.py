from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ReturnFeature:
    """Price return over a fixed number of *trading days*.

    Positive = uptrend; for reversal screening we look for *negative* values.
    """

    trading_days: int

    @property
    def name(self) -> str:
        return f"return_{self.trading_days}d"

    def compute(self, df: pd.DataFrame) -> float:
        close = df["close"].dropna()
        required = self.trading_days + 1
        if len(close) < required:
            return float("nan")
        return float(close.iloc[-1] / close.iloc[-required] - 1)


@dataclass(frozen=True)
class DailyReturnFeature:
    """Last single-day return (close[-1] / close[-2] - 1)."""

    name: str = "daily_return"

    def compute(self, df: pd.DataFrame) -> float:
        close = df["close"].dropna()
        if len(close) < 2:
            return float("nan")
        return float(close.iloc[-1] / close.iloc[-2] - 1)


class Dist52WLowFeature:
    """Distance of the latest close from the 52-week *low*.

    Formula: close[-1] / min(low[-252:]) - 1

    A value of 0.05 means the stock is 5% *above* its 52-week low.
    Lower values → closer to the 52-week low → stronger reversal candidate.
    """

    name: str = "dist_52w_low"

    def compute(self, df: pd.DataFrame) -> float:
        close = df["close"].dropna()
        low = df["low"].dropna()
        if len(close) < 2 or len(low) < 2:
            return float("nan")
        lookback = min(252, len(low))
        low_52w = float(low.iloc[-lookback:].min())
        if low_52w == 0:
            return float("nan")
        return float(close.iloc[-1] / low_52w - 1)


@dataclass(frozen=True)
class VolumeRatioFeature:
    """Ratio of today's volume to the rolling average volume.

    Formula: volume[-1] / mean(volume[-window:])

    > 1.0 means above-average volume — a key reversal signal.
    """

    window: int = 20

    @property
    def name(self) -> str:
        return f"volume_ratio_{self.window}d"

    def compute(self, df: pd.DataFrame) -> float:
        vol = df["volume"].dropna()
        if len(vol) < self.window + 1:
            return float("nan")
        avg = float(vol.iloc[-(self.window + 1) : -1].mean())
        if avg == 0:
            return float("nan")
        return float(vol.iloc[-1] / avg)


# ------------------------------------------------------------------
# Canonical singletons used by ReversalStrategy
# ------------------------------------------------------------------
Return5D = ReturnFeature(trading_days=5)
Return20D = ReturnFeature(trading_days=20)
Return60D = ReturnFeature(trading_days=60)
DailyReturn = DailyReturnFeature()
Dist52WLow = Dist52WLowFeature()
VolumeRatio20D = VolumeRatioFeature(window=20)

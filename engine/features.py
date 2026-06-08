"""engine.features — all technical feature classes and singletons."""

from features.base import Feature
from features.feature_engine import FeatureEngine
from features.momentum import Momentum3M, Momentum6M, Momentum12M, MomentumFeature
from features.oscillators import RSI14, RSIFeature
from features.reversal import (
    DailyReturn,
    DailyReturnFeature,
    Dist52WLow,
    Dist52WLowFeature,
    Return5D,
    Return20D,
    Return60D,
    ReturnFeature,
    VolumeRatio20D,
    VolumeRatioFeature,
)
from features.trend import (
    DeathCross,
    DeathCrossFeature,
    Dist52WHigh,
    Dist52WHighFeature,
    SMA50,
    SMA200,
    SMAFeature,
)
from features.volatility import Volatility30D, VolatilityFeature

__all__ = [
    "Feature",
    "FeatureEngine",
    "MomentumFeature",
    "Momentum3M",
    "Momentum6M",
    "Momentum12M",
    "RSIFeature",
    "RSI14",
    "ReturnFeature",
    "Return5D",
    "Return20D",
    "Return60D",
    "DailyReturnFeature",
    "DailyReturn",
    "Dist52WLowFeature",
    "Dist52WLow",
    "VolumeRatioFeature",
    "VolumeRatio20D",
    "SMAFeature",
    "SMA50",
    "SMA200",
    "Dist52WHighFeature",
    "Dist52WHigh",
    "DeathCrossFeature",
    "DeathCross",
    "VolatilityFeature",
    "Volatility30D",
]

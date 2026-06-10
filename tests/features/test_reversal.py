"""Tests for features/reversal.py."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from features.reversal import (
    DailyReturn,
    DailyReturnFeature,
    Dist52WLow,
    Dist52WLowFeature,
    Return20D,
    ReturnFeature,
    VolumeRatioFeature,
)


def _make_df(closes: list[float], volumes: list[float] | None = None) -> pd.DataFrame:
    n = len(closes)
    idx = pd.date_range("2023-01-02", periods=n, freq="B", name="date")
    vols = volumes or [1_000_000.0] * n
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes],
            "close": closes,
            "volume": vols,
        },
        index=idx,
    )


# ------------------------------------------------------------------
# ReturnFeature
# ------------------------------------------------------------------


class TestReturnFeature:
    def test_positive_return(self):
        df = _make_df([100.0] * 20 + [110.0])
        feat = ReturnFeature(trading_days=20)
        assert feat.compute(df) == pytest.approx(0.10)

    def test_negative_return(self):
        df = _make_df([100.0] * 20 + [90.0])
        feat = ReturnFeature(trading_days=20)
        assert feat.compute(df) == pytest.approx(-0.10)

    def test_insufficient_data_returns_nan(self):
        df = _make_df([100.0] * 5)
        feat = ReturnFeature(trading_days=20)
        assert math.isnan(feat.compute(df))

    def test_name(self):
        assert ReturnFeature(trading_days=20).name == "return_20d"
        assert ReturnFeature(trading_days=5).name == "return_5d"


# ------------------------------------------------------------------
# DailyReturnFeature
# ------------------------------------------------------------------


class TestDailyReturnFeature:
    def test_positive_day(self):
        df = _make_df([100.0, 102.0])
        assert DailyReturn.compute(df) == pytest.approx(0.02)

    def test_negative_day(self):
        df = _make_df([100.0, 95.0])
        assert DailyReturn.compute(df) == pytest.approx(-0.05)

    def test_insufficient_data_returns_nan(self):
        df = _make_df([100.0])
        assert math.isnan(DailyReturn.compute(df))

    def test_name(self):
        assert DailyReturnFeature().name == "daily_return"


# ------------------------------------------------------------------
# Dist52WLowFeature
# ------------------------------------------------------------------


class TestDist52WLowFeature:
    def test_at_52w_low(self):
        # close / low - 1: low = close * 0.99 → dist ≈ 1/0.99 - 1 ≈ 0.0101
        df = _make_df([100.0] * 252)
        val = Dist52WLow.compute(df)
        assert val == pytest.approx(100.0 / 99.0 - 1, rel=1e-4)

    def test_above_52w_low(self):
        prices = [100.0] * 251 + [110.0]
        df = _make_df(prices)
        # 52w low ≈ 99.0 (low = close * 0.99), close = 110
        # dist = 110 / 99.0 - 1 ≈ 0.111
        val = Dist52WLow.compute(df)
        assert val > 0.0

    def test_insufficient_data_returns_nan(self):
        df = _make_df([100.0])
        assert math.isnan(Dist52WLow.compute(df))

    def test_name(self):
        assert Dist52WLowFeature().name == "dist_52w_low"


# ------------------------------------------------------------------
# VolumeRatioFeature
# ------------------------------------------------------------------


class TestVolumeRatioFeature:
    def test_volume_above_average(self):
        vols = [1_000_000.0] * 20 + [2_000_000.0]
        df = _make_df([100.0] * 21, volumes=vols)
        feat = VolumeRatioFeature(window=20)
        assert feat.compute(df) == pytest.approx(2.0)

    def test_volume_below_average(self):
        vols = [1_000_000.0] * 20 + [500_000.0]
        df = _make_df([100.0] * 21, volumes=vols)
        feat = VolumeRatioFeature(window=20)
        assert feat.compute(df) == pytest.approx(0.5)

    def test_insufficient_data_returns_nan(self):
        df = _make_df([100.0] * 5)
        feat = VolumeRatioFeature(window=20)
        assert math.isnan(feat.compute(df))

    def test_name(self):
        assert VolumeRatioFeature(window=20).name == "volume_ratio_20d"
        assert VolumeRatioFeature(window=5).name == "volume_ratio_5d"

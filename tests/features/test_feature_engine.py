from __future__ import annotations

import math
from datetime import date

import pandas as pd
import pytest

from features.feature_engine import FeatureEngine
from features.momentum import Momentum3M, Momentum6M
from features.oscillators import RSI14
from features.trend import SMA50


# ------------------------------------------------------------------
# FeatureEngine.default()
# ------------------------------------------------------------------


def test_default_engine_has_expected_features() -> None:
    engine = FeatureEngine.default()
    names = engine.feature_names
    for expected in [
        "momentum_3m",
        "momentum_6m",
        "momentum_12m",
        "sma_50",
        "sma_200",
        "dist_52w_high",
        "death_cross",
        "rsi_14",
        "volatility_30d",
    ]:
        assert expected in names


# ------------------------------------------------------------------
# compute_row
# ------------------------------------------------------------------


def test_compute_row_returns_all_feature_keys(long_price_df: pd.DataFrame) -> None:
    engine = FeatureEngine.default()
    row = engine.compute_row("AAPL", long_price_df)
    assert row["ticker"] == "AAPL"
    for name in engine.feature_names:
        assert name in row


def test_compute_row_no_nan_on_sufficient_data(long_price_df: pd.DataFrame) -> None:
    engine = FeatureEngine.default()
    row = engine.compute_row("AAPL", long_price_df)
    for key, val in row.items():
        if key == "ticker":
            continue
        assert not math.isnan(val), f"Feature '{key}' returned NaN"


def test_compute_row_returns_nan_on_short_data(short_price_df: pd.DataFrame) -> None:
    engine = FeatureEngine([Momentum3M, RSI14])
    row = engine.compute_row("AAPL", short_price_df)
    assert math.isnan(row["momentum_3m"])
    assert math.isnan(row["rsi_14"])


# ------------------------------------------------------------------
# compute_all
# ------------------------------------------------------------------


def test_compute_all_returns_dataframe(long_price_df: pd.DataFrame) -> None:
    engine = FeatureEngine([SMA50, RSI14])
    result = engine.compute_all({"AAPL": long_price_df, "MSFT": long_price_df})
    assert isinstance(result, pd.DataFrame)
    assert set(result.index) == {"AAPL", "MSFT"}
    assert "sma_50" in result.columns
    assert "rsi_14" in result.columns


def test_compute_all_empty_input_returns_empty_df() -> None:
    engine = FeatureEngine.default()
    result = engine.compute_all({})
    assert result.empty


def test_compute_all_index_is_ticker(long_price_df: pd.DataFrame) -> None:
    engine = FeatureEngine([SMA50])
    result = engine.compute_all({"NVDA": long_price_df})
    assert result.index.name == "ticker"
    assert "NVDA" in result.index


# ------------------------------------------------------------------
# as_of (backtest mode)
# ------------------------------------------------------------------


def test_compute_all_as_of_slices_data(long_price_df: pd.DataFrame) -> None:
    engine = FeatureEngine([Momentum3M])
    as_of = date(2023, 6, 1)

    result_full = engine.compute_all({"AAPL": long_price_df})
    result_sliced = engine.compute_all({"AAPL": long_price_df}, as_of=as_of)

    # Result with as_of should differ from the full-history result
    assert result_full.loc["AAPL", "momentum_3m"] != pytest.approx(
        result_sliced.loc["AAPL", "momentum_3m"]
    )


def test_compute_all_as_of_does_not_use_future_data(
    long_price_df: pd.DataFrame,
) -> None:
    engine = FeatureEngine([SMA50])
    as_of = date(2023, 4, 1)

    # Compute manually by slicing
    manual_df = long_price_df.loc[: str(as_of)]
    expected = SMA50.compute(manual_df)

    result = engine.compute_all({"AAPL": long_price_df}, as_of=as_of)
    assert result.loc["AAPL", "sma_50"] == pytest.approx(expected)

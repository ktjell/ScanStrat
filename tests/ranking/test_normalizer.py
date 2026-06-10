from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from ranking.normalizer import Normalizer


# ------------------------------------------------------------------
# Basic output shape
# ------------------------------------------------------------------


def test_normalize_same_shape(feature_df: pd.DataFrame) -> None:
    result = Normalizer().normalize(feature_df)
    assert result.shape == feature_df.shape
    assert list(result.columns) == list(feature_df.columns)
    assert list(result.index) == list(feature_df.index)


def test_normalize_values_in_0_100(feature_df: pd.DataFrame) -> None:
    result = Normalizer().normalize(feature_df)
    assert (result.dropna() >= 0).all().all()
    assert (result.dropna() <= 100).all().all()


# ------------------------------------------------------------------
# Ordering: best raw value → highest percentile score
# ------------------------------------------------------------------


def test_ascending_feature_highest_raw_gets_highest_score(
    feature_df: pd.DataFrame,
) -> None:
    result = Normalizer().normalize(feature_df)
    # NVDA has highest momentum_12m → should get highest score
    assert result["momentum_12m"].idxmax() == "NVDA"
    assert result["momentum_12m"].idxmin() == "TSLA"


def test_descending_feature_lowest_raw_gets_highest_score(
    feature_df: pd.DataFrame,
) -> None:
    # volatility_30d is descending: low vol → high score
    result = Normalizer().normalize(feature_df)
    assert result["volatility_30d"].idxmax() == "NVDA"  # lowest raw vol
    assert result["volatility_30d"].idxmin() == "TSLA"  # highest raw vol


# ------------------------------------------------------------------
# NaN propagation
# ------------------------------------------------------------------


def test_nan_in_input_propagates_to_output(feature_df: pd.DataFrame) -> None:
    df = feature_df.copy()
    df.loc["AAPL", "momentum_12m"] = float("nan")
    result = Normalizer().normalize(df)
    assert math.isnan(result.loc["AAPL", "momentum_12m"])
    # Other tickers must still have valid scores
    assert not math.isnan(result.loc["NVDA", "momentum_12m"])


# ------------------------------------------------------------------
# Single ticker edge case
# ------------------------------------------------------------------


def test_single_ticker_gets_score_100(feature_df: pd.DataFrame) -> None:
    single = feature_df.iloc[:1]
    result = Normalizer().normalize(single)
    # With one ticker, it ranks at 100th percentile for all features
    assert (result.dropna() == 100.0).all().all()

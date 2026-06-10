from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ranking.scorer import Scorer


_WEIGHTS = {
    "momentum_12m": 0.25,
    "momentum_6m": 0.20,
    "momentum_3m": 0.15,
    "dist_52w_high": 0.15,
    "rsi_14": 0.10,
    "volatility_30d": 0.15,
}


@pytest.fixture
def normalised_df(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Dummy normalised scores (0–100) matching feature_df tickers."""
    data = {col: [80.0, 70.0, 50.0, 30.0, 10.0] for col in feature_df.columns}
    return pd.DataFrame(data, index=feature_df.index)


# ------------------------------------------------------------------
# Basic output
# ------------------------------------------------------------------


def test_scorer_returns_series(normalised_df: pd.DataFrame) -> None:
    scorer = Scorer(_WEIGHTS)
    result = scorer.score(normalised_df)
    assert isinstance(result, pd.Series)
    assert result.name == "score"


def test_scorer_index_matches_input(normalised_df: pd.DataFrame) -> None:
    scorer = Scorer(_WEIGHTS)
    result = scorer.score(normalised_df)
    assert list(result.index) == list(normalised_df.index)


def test_scores_in_0_100_range(normalised_df: pd.DataFrame) -> None:
    scorer = Scorer(_WEIGHTS)
    result = scorer.score(normalised_df)
    assert (result >= 0).all()
    assert (result <= 100).all()


# ------------------------------------------------------------------
# Ordering
# ------------------------------------------------------------------


def test_highest_normalised_gets_highest_score(normalised_df: pd.DataFrame) -> None:
    scorer = Scorer(_WEIGHTS)
    result = scorer.score(normalised_df)
    assert result.idxmax() == "NVDA"
    assert result.idxmin() == "TSLA"


# ------------------------------------------------------------------
# NaN handling: filled with neutral 50
# ------------------------------------------------------------------


def test_nan_filled_with_neutral(normalised_df: pd.DataFrame) -> None:
    df = normalised_df.copy()
    df.loc["AAPL", "momentum_12m"] = float("nan")
    scorer = Scorer(_WEIGHTS)
    result = scorer.score(df)
    assert not result.isna().any()


# ------------------------------------------------------------------
# Weights are normalised to sum to 1
# ------------------------------------------------------------------


def test_weights_normalisation() -> None:
    # Pass weights that don't sum to 1
    scorer = Scorer({"a": 2.0, "b": 2.0})
    df = pd.DataFrame({"a": [100.0], "b": [100.0]}, index=["X"])
    result = scorer.score(df)
    assert result["X"] == pytest.approx(100.0)


# ------------------------------------------------------------------
# Unknown feature columns are silently ignored
# ------------------------------------------------------------------


def test_unknown_feature_is_ignored(normalised_df: pd.DataFrame) -> None:
    weights = dict(_WEIGHTS)
    weights["nonexistent_feature"] = 0.99
    scorer = Scorer(weights)
    result = scorer.score(normalised_df)
    assert not result.isna().any()


# ------------------------------------------------------------------
# Empty weights raises
# ------------------------------------------------------------------


def test_empty_weights_raises() -> None:
    with pytest.raises(ValueError, match="weights must not be empty"):
        Scorer({})

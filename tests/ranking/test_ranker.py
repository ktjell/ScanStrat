from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from config.settings import RankingSettings, Settings
from ranking.normalizer import Normalizer
from ranking.ranker import Ranker
from ranking.scorer import Scorer
from features.feature_engine import FeatureEngine


_WEIGHTS = {
    "momentum_12m": 0.25,
    "momentum_6m": 0.20,
    "momentum_3m": 0.15,
    "dist_52w_high": 0.15,
    "rsi_14": 0.10,
    "volatility_30d": 0.15,
}


@pytest.fixture
def ranker(feature_df: pd.DataFrame) -> Ranker:
    """Ranker with a mocked FeatureEngine that returns feature_df."""
    mock_engine = MagicMock(spec=FeatureEngine)
    mock_engine.compute_all.return_value = feature_df
    return Ranker(
        feature_engine=mock_engine,
        normalizer=Normalizer(),
        scorer=Scorer(_WEIGHTS),
    )


# ------------------------------------------------------------------
# Output structure
# ------------------------------------------------------------------


def test_rank_returns_dataframe(ranker: Ranker, feature_df: pd.DataFrame) -> None:
    data = {t: pd.DataFrame() for t in feature_df.index}
    result = ranker.rank(data)
    assert isinstance(result, pd.DataFrame)


def test_rank_has_rank_ticker_score_columns(
    ranker: Ranker, feature_df: pd.DataFrame
) -> None:
    data = {t: pd.DataFrame() for t in feature_df.index}
    result = ranker.rank(data)
    assert "rank" in result.columns
    assert "ticker" in result.columns
    assert "score" in result.columns


def test_rank_number_of_rows_matches_tickers(
    ranker: Ranker, feature_df: pd.DataFrame
) -> None:
    data = {t: pd.DataFrame() for t in feature_df.index}
    result = ranker.rank(data)
    assert len(result) == len(feature_df)


# ------------------------------------------------------------------
# Ordering
# ------------------------------------------------------------------


def test_rank_1_has_highest_score(ranker: Ranker, feature_df: pd.DataFrame) -> None:
    data = {t: pd.DataFrame() for t in feature_df.index}
    result = ranker.rank(data)
    assert result.iloc[0]["rank"] == 1
    assert result["score"].iloc[0] == result["score"].max()


def test_rank_sorted_descending_by_score(
    ranker: Ranker, feature_df: pd.DataFrame
) -> None:
    data = {t: pd.DataFrame() for t in feature_df.index}
    result = ranker.rank(data)
    assert result["score"].is_monotonic_decreasing


def test_nvda_ranks_first(ranker: Ranker, feature_df: pd.DataFrame) -> None:
    """NVDA has the best features in the fixture → should be rank 1."""
    data = {t: pd.DataFrame() for t in feature_df.index}
    result = ranker.rank(data)
    assert result.iloc[0]["ticker"] == "NVDA"


def test_tsla_ranks_last(ranker: Ranker, feature_df: pd.DataFrame) -> None:
    data = {t: pd.DataFrame() for t in feature_df.index}
    result = ranker.rank(data)
    assert result.iloc[-1]["ticker"] == "TSLA"


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------


def test_empty_data_returns_empty_df(ranker: Ranker) -> None:
    ranker._feature_engine.compute_all.return_value = pd.DataFrame()
    result = ranker.rank({})
    assert result.empty


def test_rank_passes_as_of_to_feature_engine(
    ranker: Ranker, feature_df: pd.DataFrame
) -> None:
    data = {t: pd.DataFrame() for t in feature_df.index}
    as_of = date(2024, 6, 1)
    ranker.rank(data, as_of=as_of)
    ranker._feature_engine.compute_all.assert_called_once_with(data, as_of=as_of)


# ------------------------------------------------------------------
# Death cross filter
# ------------------------------------------------------------------


def test_death_cross_tickers_are_excluded(feature_df: pd.DataFrame) -> None:
    """Tickers with death_cross == 1.0 must not appear in output."""
    df = feature_df.copy()
    df["death_cross"] = 0.0
    df.loc["TSLA", "death_cross"] = 1.0  # TSLA in active death cross

    mock_engine = MagicMock(spec=FeatureEngine)
    mock_engine.compute_all.return_value = df
    r = Ranker(
        feature_engine=mock_engine, normalizer=Normalizer(), scorer=Scorer(_WEIGHTS)
    )

    result = r.rank({t: pd.DataFrame() for t in df.index})
    assert "TSLA" not in result["ticker"].values
    assert len(result) == 4


def test_all_death_cross_returns_empty(feature_df: pd.DataFrame) -> None:
    """If every ticker is in death cross the result is empty."""
    df = feature_df.copy()
    df["death_cross"] = 1.0

    mock_engine = MagicMock(spec=FeatureEngine)
    mock_engine.compute_all.return_value = df
    r = Ranker(
        feature_engine=mock_engine, normalizer=Normalizer(), scorer=Scorer(_WEIGHTS)
    )

    result = r.rank({t: pd.DataFrame() for t in df.index})
    assert result.empty


def test_no_death_cross_column_keeps_all(
    ranker: Ranker, feature_df: pd.DataFrame
) -> None:
    """If death_cross is absent (old data), all tickers are kept."""
    result = ranker.rank({t: pd.DataFrame() for t in feature_df.index})
    assert len(result) == len(feature_df)


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------


def test_default_factory_builds_ranker() -> None:
    settings = Settings.default()
    ranker = Ranker.default(settings.ranking)
    assert isinstance(ranker, Ranker)

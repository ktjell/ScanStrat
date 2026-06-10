"""Tests for ranking/reversal_strategy.py."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from config.settings import ReversalSettings
from ranking.reversal_strategy import ReversalStrategy


def _make_df(
    closes: list[float], volumes: list[float] | None = None, n_extra: int = 0
) -> pd.DataFrame:
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


def _reversal_candidate(n: int = 260) -> pd.DataFrame:
    """A stock that should pass all reversal filters:
    - Declining price (negative 20d return)
    - Low RSI (oversold)
    - High volume on last day
    - Near 52w low
    """
    # Declining prices
    closes = [100.0 - i * 0.15 for i in range(n)]
    # Last day: small uptick
    closes[-1] = closes[-2] * 1.005
    # High volume on last day
    volumes = [1_000_000.0] * n
    volumes[-1] = 2_500_000.0
    return _make_df(closes, volumes=volumes)


def _strong_uptrend(n: int = 260) -> pd.DataFrame:
    """A stock in strong uptrend — should NOT pass reversal filters."""
    closes = [100.0 + i * 0.5 for i in range(n)]
    volumes = [1_000_000.0] * n
    return _make_df(closes, volumes=volumes)


class TestReversalStrategyFilters:
    def test_empty_data_returns_empty(self):
        strategy = ReversalStrategy.default()
        result = strategy.rank({}, as_of=None)
        assert result.empty

    def test_uptrend_stock_excluded(self):
        """A stock in strong uptrend should not pass the negative-return filter."""
        strategy = ReversalStrategy(
            ReversalSettings(
                rsi_threshold=35.0,
                volume_ratio_threshold=1.2,
                return_lookback_days=20,
                max_dist_52w_low=0.25,
            )
        )
        data = {"STRONG": _strong_uptrend()}
        result = strategy.rank(data)
        assert result.empty

    def test_reversal_candidate_can_pass(self):
        """A declining stock with high volume can pass filters if RSI is low enough."""
        strategy = ReversalStrategy(
            ReversalSettings(
                rsi_threshold=50.0,  # Lenient threshold
                volume_ratio_threshold=1.5,
                return_lookback_days=20,
                max_dist_52w_low=None,  # Disable proximity filter
                require_positive_day=False,
            )
        )
        data = {"CAND": _reversal_candidate()}
        result = strategy.rank(data)
        assert not result.empty
        assert result.iloc[0]["ticker"] == "CAND"

    def test_output_columns_present(self):
        strategy = ReversalStrategy(
            ReversalSettings(
                rsi_threshold=50.0,
                volume_ratio_threshold=1.5,
                return_lookback_days=20,
                max_dist_52w_low=None,
                require_positive_day=False,
            )
        )
        data = {"CAND": _reversal_candidate()}
        result = strategy.rank(data)
        if result.empty:
            pytest.skip("No candidates passed filters with test data")
        assert "rank" in result.columns
        assert "reversal_score" in result.columns
        assert "ticker" in result.columns

    def test_rank_column_starts_at_1(self):
        strategy = ReversalStrategy(
            ReversalSettings(
                rsi_threshold=50.0,
                volume_ratio_threshold=1.0,
                return_lookback_days=20,
                max_dist_52w_low=None,
                require_positive_day=False,
            )
        )
        data = {
            "A": _reversal_candidate(),
            "B": _reversal_candidate(),
        }
        result = strategy.rank(data)
        if result.empty:
            pytest.skip("No candidates passed")
        assert result["rank"].iloc[0] == 1

    def test_scores_descending(self):
        strategy = ReversalStrategy(
            ReversalSettings(
                rsi_threshold=50.0,
                volume_ratio_threshold=1.0,
                return_lookback_days=20,
                max_dist_52w_low=None,
                require_positive_day=False,
            )
        )
        data = {
            "A": _reversal_candidate(),
            "B": _reversal_candidate(),
            "C": _reversal_candidate(),
        }
        result = strategy.rank(data)
        if len(result) < 2:
            pytest.skip("Need at least 2 candidates")
        scores = result["reversal_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_require_positive_day_filter(self):
        """With require_positive_day=True, a stock with negative last day is excluded."""
        strategy = ReversalStrategy(
            ReversalSettings(
                rsi_threshold=50.0,
                volume_ratio_threshold=1.0,
                return_lookback_days=20,
                max_dist_52w_low=None,
                require_positive_day=True,
            )
        )
        # Make a stock with negative last day
        closes = [100.0 - i * 0.15 for i in range(260)]
        closes[-1] = closes[-2] * 0.99  # negative day
        volumes = [1_000_000.0] * 260
        volumes[-1] = 3_000_000.0
        data = {"NEG_DAY": _make_df(closes, volumes=volumes)}
        result = strategy.rank(data)
        assert result.empty


class TestReversalStrategyDefault:
    def test_default_factory(self):
        strategy = ReversalStrategy.default()
        assert isinstance(strategy, ReversalStrategy)

    def test_custom_settings(self):
        settings = ReversalSettings(rsi_threshold=30.0, volume_ratio_threshold=2.0)
        strategy = ReversalStrategy(settings)
        assert strategy._s.rsi_threshold == 30.0
        assert strategy._s.volume_ratio_threshold == 2.0

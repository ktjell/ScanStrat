from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from backtest.engine import BacktestEngine, BacktestResult
from ranking.ranker import Ranker
from tests.backtest.conftest import make_price_df, make_rising_df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_ranker(tickers: list[str]) -> MagicMock:
    """Ranker that always returns the same fixed ranking."""
    mock = MagicMock(spec=Ranker)
    mock.rank.return_value = pd.DataFrame(
        {
            "rank": range(1, len(tickers) + 1),
            "ticker": tickers,
            "score": range(100, 100 - len(tickers), -1),
        }
    )
    return mock


def _flat_data(
    tickers: list[str], start: str = "2022-01-01", periods: int = 500
) -> dict[str, pd.DataFrame]:
    return {t: make_price_df(start, periods) for t in tickers}


# ---------------------------------------------------------------------------
# BacktestEngine construction
# ---------------------------------------------------------------------------


def test_top_n_zero_raises() -> None:
    with pytest.raises(ValueError):
        BacktestEngine(ranker=_mock_ranker(["A"]), top_n=0)


def test_top_n_negative_raises() -> None:
    with pytest.raises(ValueError):
        BacktestEngine(ranker=_mock_ranker(["A"]), top_n=-1)


# ---------------------------------------------------------------------------
# run — output structure
# ---------------------------------------------------------------------------


def test_run_returns_backtest_result() -> None:
    tickers = ["AAPL", "MSFT", "NVDA"]
    engine = BacktestEngine(ranker=_mock_ranker(tickers), top_n=2, rebalance_freq="ME")
    result = engine.run(
        _flat_data(tickers), start=date(2022, 1, 1), end=date(2022, 6, 30)
    )
    assert isinstance(result, BacktestResult)


def test_equity_curve_starts_at_one() -> None:
    tickers = ["AAPL", "MSFT"]
    engine = BacktestEngine(ranker=_mock_ranker(tickers), top_n=2, rebalance_freq="ME")
    result = engine.run(
        _flat_data(tickers), start=date(2022, 1, 1), end=date(2022, 6, 30)
    )
    assert result.equity_curve.iloc[0] == pytest.approx(1.0)


def test_equity_curve_is_series() -> None:
    tickers = ["AAPL", "MSFT"]
    engine = BacktestEngine(ranker=_mock_ranker(tickers), top_n=2, rebalance_freq="ME")
    result = engine.run(
        _flat_data(tickers), start=date(2022, 1, 1), end=date(2022, 6, 30)
    )
    assert isinstance(result.equity_curve, pd.Series)


def test_holdings_is_dataframe() -> None:
    tickers = ["AAPL", "MSFT"]
    engine = BacktestEngine(ranker=_mock_ranker(tickers), top_n=2, rebalance_freq="ME")
    result = engine.run(
        _flat_data(tickers), start=date(2022, 1, 1), end=date(2022, 6, 30)
    )
    assert isinstance(result.holdings, pd.DataFrame)


def test_metrics_has_required_keys() -> None:
    tickers = ["AAPL", "MSFT"]
    engine = BacktestEngine(ranker=_mock_ranker(tickers), top_n=2, rebalance_freq="ME")
    result = engine.run(
        _flat_data(tickers), start=date(2022, 1, 1), end=date(2022, 6, 30)
    )
    assert {"cagr", "sharpe", "max_drawdown", "total_return"} <= set(
        result.metrics.keys()
    )


# ---------------------------------------------------------------------------
# run — correctness with flat prices
# ---------------------------------------------------------------------------


def test_flat_prices_zero_total_return() -> None:
    """Flat prices → no gain, no loss."""
    tickers = ["AAPL", "MSFT"]
    engine = BacktestEngine(ranker=_mock_ranker(tickers), top_n=2, rebalance_freq="ME")
    result = engine.run(
        _flat_data(tickers), start=date(2022, 1, 1), end=date(2022, 6, 30)
    )
    assert result.metrics["total_return"] == pytest.approx(0.0, abs=1e-6)


def test_rising_prices_positive_total_return() -> None:
    """Rising prices → positive total return."""
    tickers = ["AAPL", "MSFT"]
    data = {
        t: make_rising_df("2022-01-01", periods=400, start_price=100, end_price=200)
        for t in tickers
    }
    engine = BacktestEngine(ranker=_mock_ranker(tickers), top_n=2, rebalance_freq="ME")
    result = engine.run(data, start=date(2022, 1, 1), end=date(2022, 12, 31))
    assert result.metrics["total_return"] > 0


# ---------------------------------------------------------------------------
# run — date range too short
# ---------------------------------------------------------------------------


def test_too_short_raises() -> None:
    tickers = ["AAPL"]
    engine = BacktestEngine(ranker=_mock_ranker(tickers), top_n=1, rebalance_freq="ME")
    with pytest.raises(ValueError):
        engine.run(_flat_data(tickers), start=date(2022, 1, 1), end=date(2022, 1, 15))


# ---------------------------------------------------------------------------
# run — top_n capped by available tickers
# ---------------------------------------------------------------------------


def test_top_n_capped_to_available_tickers() -> None:
    """Ask for top-10 but only 2 pass the ranker — should not crash."""
    tickers = ["AAPL", "MSFT"]
    engine = BacktestEngine(ranker=_mock_ranker(tickers), top_n=10, rebalance_freq="ME")
    result = engine.run(
        _flat_data(tickers), start=date(2022, 1, 1), end=date(2022, 6, 30)
    )
    # Holdings should show at most 2 per period
    assert (result.holdings["n_held"] <= 2).all()


# ---------------------------------------------------------------------------
# commission
# ---------------------------------------------------------------------------


def test_negative_commission_raises() -> None:
    with pytest.raises(ValueError):
        BacktestEngine(ranker=_mock_ranker(["AAPL"]), top_n=1, portfolio_size_usd=-1.0)


def test_commission_reduces_return_vs_no_commission() -> None:
    """With rising prices, adding commission must lower total return."""
    from backtest.commission import CommissionSchedule

    tickers = ["AAPL", "MSFT"]
    data = {t: make_rising_df("2022-01-01", periods=400) for t in tickers}

    no_cost = BacktestEngine(
        ranker=_mock_ranker(tickers),
        top_n=2,
        rebalance_freq="ME",
        commission_schedule=CommissionSchedule.zero(),
    )
    with_cost = BacktestEngine(
        ranker=_mock_ranker(tickers),
        top_n=2,
        rebalance_freq="ME",
        commission_schedule=CommissionSchedule.saxo_classic(),
    )

    r_no = no_cost.run(data, start=date(2022, 1, 1), end=date(2022, 12, 31))
    r_with = with_cost.run(data, start=date(2022, 1, 1), end=date(2022, 12, 31))

    assert r_with.metrics["total_return"] < r_no.metrics["total_return"]


def test_flat_prices_with_commission_gives_negative_return() -> None:
    """Flat prices + commission = guaranteed loss per period."""
    from backtest.commission import CommissionSchedule

    tickers = ["AAPL"]
    engine = BacktestEngine(
        ranker=_mock_ranker(tickers),
        top_n=1,
        rebalance_freq="ME",
        commission_schedule=CommissionSchedule.saxo_classic(),
    )
    result = engine.run(
        _flat_data(tickers), start=date(2022, 1, 1), end=date(2022, 6, 30)
    )
    assert result.metrics["total_return"] < 0

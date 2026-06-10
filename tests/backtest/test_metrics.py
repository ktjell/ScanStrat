from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from backtest.metrics import cagr, sharpe, max_drawdown, win_rate, compute_metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _equity(values: list[float], start: str = "2023-01-01") -> pd.Series:
    idx = pd.date_range(start, periods=len(values), freq="D")
    return pd.Series(values, index=idx, name="portfolio_value")


# ---------------------------------------------------------------------------
# cagr
# ---------------------------------------------------------------------------


def test_cagr_flat_returns_zero() -> None:
    eq = _equity([1.0, 1.0, 1.0], start="2020-01-01")
    assert cagr(eq) == pytest.approx(0.0, abs=1e-6)


def test_cagr_doubles_in_one_year() -> None:
    # Build exactly one year of daily data ending at 2.0
    idx = pd.date_range("2020-01-01", "2021-01-01", freq="D")
    values = np.linspace(1.0, 2.0, len(idx))
    eq = pd.Series(values, index=idx)
    result = cagr(eq)
    assert result == pytest.approx(1.0, abs=0.02)  # ~100% CAGR


def test_cagr_too_short_returns_nan() -> None:
    assert math.isnan(cagr(_equity([1.0])))


def test_cagr_zero_years_returns_nan() -> None:
    eq = _equity([1.0, 1.5])  # two points same day after freq="D" — force same day
    eq.index = pd.DatetimeIndex(["2020-01-01", "2020-01-01"])
    assert math.isnan(cagr(eq))


# ---------------------------------------------------------------------------
# sharpe
# ---------------------------------------------------------------------------


def test_sharpe_flat_equity_returns_nan() -> None:
    eq = _equity([1.0] * 50)
    assert math.isnan(sharpe(eq))


def test_sharpe_rising_equity_is_positive() -> None:
    idx = pd.date_range("2020-01-01", periods=252, freq="D")
    values = np.linspace(1.0, 1.5, 252)
    eq = pd.Series(values, index=idx)
    assert sharpe(eq) > 0


def test_sharpe_too_short_returns_nan() -> None:
    assert math.isnan(sharpe(_equity([1.0])))


# ---------------------------------------------------------------------------
# max_drawdown
# ---------------------------------------------------------------------------


def test_max_drawdown_no_drawdown() -> None:
    eq = _equity([1.0, 1.1, 1.2, 1.3])
    assert max_drawdown(eq) == pytest.approx(0.0, abs=1e-6)


def test_max_drawdown_50_percent() -> None:
    eq = _equity([1.0, 2.0, 1.0])
    assert max_drawdown(eq) == pytest.approx(-0.5, abs=1e-6)


def test_max_drawdown_too_short_returns_nan() -> None:
    assert math.isnan(max_drawdown(_equity([1.0])))


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# win_rate
# ---------------------------------------------------------------------------


def test_win_rate_all_up() -> None:
    eq = _equity([1.0, 1.1, 1.2, 1.3])
    assert win_rate(eq) == pytest.approx(1.0)


def test_win_rate_all_down() -> None:
    eq = _equity([1.3, 1.2, 1.1, 1.0])
    assert win_rate(eq) == pytest.approx(0.0)


def test_win_rate_half() -> None:
    eq = _equity([1.0, 1.1, 1.0, 1.1])
    assert win_rate(eq) == pytest.approx(2 / 3)  # 2 op af 3 daglige afkast


def test_win_rate_too_short_returns_nan() -> None:
    assert math.isnan(win_rate(_equity([1.0])))


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------


def test_compute_metrics_returns_all_keys() -> None:
    idx = pd.date_range("2020-01-01", periods=252, freq="D")
    eq = pd.Series(np.linspace(1.0, 1.4, 252), index=idx)
    m = compute_metrics(eq)
    assert set(m.keys()) == {
        "cagr",
        "sharpe",
        "max_drawdown",
        "total_return",
        "win_rate",
    }


def test_compute_metrics_total_return() -> None:
    eq = _equity([1.0, 1.5])
    m = compute_metrics(eq)
    assert m["total_return"] == pytest.approx(0.5)

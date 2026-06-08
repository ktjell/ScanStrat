from __future__ import annotations

import math

import numpy as np
import pandas as pd


def cagr(equity: pd.Series) -> float:
    """Compound Annual Growth Rate.

    Parameters
    ----------
    equity:
        Portfolio value time series with a DatetimeIndex.
    """
    if len(equity) < 2:
        return float("nan")
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0:
        return float("nan")
    return (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0


def sharpe(equity: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualised Sharpe ratio from a daily equity curve."""
    daily_returns = equity.pct_change().dropna()
    if len(daily_returns) < 2:
        return float("nan")
    excess = daily_returns - risk_free_rate / 252
    std = excess.std()
    if std == 0:
        return float("nan")
    return float(excess.mean() / std * math.sqrt(252))


def max_drawdown(equity: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (negative number, e.g. -0.15 = -15%)."""
    if len(equity) < 2:
        return float("nan")
    rolling_max = equity.cummax()
    drawdowns = equity / rolling_max - 1.0
    return float(drawdowns.min())


def win_rate(equity: pd.Series) -> float:
    """Fraction of periods (days) with a positive return.

    Returns a value in [0, 1], e.g. 0.55 = 55% of days were up.
    """
    daily_returns = equity.pct_change().dropna()
    if len(daily_returns) == 0:
        return float("nan")
    return float((daily_returns > 0).sum() / len(daily_returns))


def compute_metrics(equity: pd.Series, risk_free_rate: float = 0.0) -> dict[str, float]:
    """Return all summary metrics for an equity curve."""
    return {
        "cagr": cagr(equity),
        "sharpe": sharpe(equity, risk_free_rate),
        "max_drawdown": max_drawdown(equity),
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0)
        if len(equity) >= 2
        else float("nan"),
        "win_rate": win_rate(equity),
    }

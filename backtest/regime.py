"""backtest/regime.py

Markedsregime-klassificering.

Regime bestemmes ud fra benchmark (SPY/IWDA) relative til SMA200:
  BULL    — benchmark > SMA200  (trending opad)
  BEAR    — benchmark < SMA200  (trending nedad)

Bruges til:
  - Farvede baggrundszoner i equity-curve plot
  - Per-regime metrics tabel: hvilke strategier virker bedst i bull/bear?
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def classify(
    bm_close: pd.Series,
    sma_period: int = 200,
) -> pd.Series:
    """Returner en dato-indekseret Series med regime-label per dag.

    Vaerdier: "bull" | "bear"
    Index: DatetimeIndex (samme som bm_close)
    """
    bm_close = bm_close.copy()
    bm_close.index = pd.to_datetime(bm_close.index)
    sma = bm_close.rolling(sma_period).mean()
    regime = pd.Series(
        np.where(bm_close >= sma, "bull", "bear"),
        index=bm_close.index,
        name="regime",
    )
    # Foerste sma_period dage er NaN — saet som bull (ikke nok data)
    regime[sma.isna()] = "bull"
    return regime


def regime_periods(regime: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    """Returner en liste af (start, slut, regime) for sammenhængende perioder."""
    periods: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
    if regime.empty:
        return periods

    current = regime.iloc[0]
    seg_start = regime.index[0]

    for ts, val in regime.items():
        if val != current:
            periods.append((seg_start, ts, current))
            seg_start = ts
            current = val

    periods.append((seg_start, regime.index[-1], current))
    return periods


def regime_metrics(
    equity: pd.Series,
    regime: pd.Series,
) -> dict[str, dict[str, float]]:
    """Beregn CAGR og max drawdown per regime for én equity curve.

    Returnerer {"bull": {"cagr": ..., "max_drawdown": ...}, "bear": {...}}
    """
    equity = equity.copy()
    equity.index = pd.to_datetime(equity.index)
    regime = regime.reindex(equity.index, method="ffill")

    result: dict[str, dict[str, float]] = {}
    for label in ("bull", "bear"):
        mask = regime == label
        sub = equity[mask]
        if len(sub) < 5:
            result[label] = {
                "cagr": float("nan"),
                "max_drawdown": float("nan"),
                "total_return": float("nan"),
            }
            continue

        days = (sub.index[-1] - sub.index[0]).days
        years = days / 365.25
        total = float(sub.iloc[-1] / sub.iloc[0] - 1)
        cagr = float((1 + total) ** (1 / years) - 1) if years > 0.1 else float("nan")
        dd = float((sub / sub.cummax() - 1).min())
        result[label] = {"cagr": cagr, "max_drawdown": dd, "total_return": total}

    return result

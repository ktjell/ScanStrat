"""research/strategies/weekly_momentum.py

Weekly Momentum: ugentlig rebalancering baseret paa kort prisimpuls.

Score = vægtet kombination af:
  - 4-ugers afkast  (20 handelsdage)  — vaegt 0.4
  - 8-ugers afkast  (40 handelsdage)  — vaegt 0.3
  - 12-ugers afkast (60 handelsdage)  — vaegt 0.3

Logik: vi vil ride trends der allerede er i gang de seneste 1-3 maaneder,
ikke vente 12 maaneder paa at bekrafte momentum.

Filter: kun aktier over SMA50 (uptrend — undgaar faldende knive).

Vectoriseret: score-matricen (dato x ticker) beregnes ét slag ved foerste kald.
Alle efterfolgende kald er O(1) row-lookups (samme teknink som DailyBreakout).
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Parametre
# ------------------------------------------------------------------
W1: int = 20  # 4 uger
W2: int = 40  # 8 uger
W3: int = 60  # 12 uger
WEIGHT1: float = 0.4
WEIGHT2: float = 0.3
WEIGHT3: float = 0.3
SMA_FILTER: int = 50  # kun aktier over SMA50

REBALANCE_DAYS: int = 5  # ugentlig
TOP_N: int = 10


class WeeklyMomentumStrategy:
    """Vectoriseret ugentlig momentum-strategi.

    Foerste kald til rank() beregner hele score-matricen (dato x ticker).
    Efterfølgende kald bruger matrix-opslag.
    """

    def __init__(
        self,
        w1: int = W1,
        w2: int = W2,
        w3: int = W3,
        weight1: float = WEIGHT1,
        weight2: float = WEIGHT2,
        weight3: float = WEIGHT3,
        sma_filter: int = SMA_FILTER,
    ) -> None:
        self._w1 = w1
        self._w2 = w2
        self._w3 = w3
        self._wt1 = weight1
        self._wt2 = weight2
        self._wt3 = weight3
        self._sma = sma_filter
        self._score_matrix: pd.DataFrame | None = None
        self._trend_matrix: pd.DataFrame | None = None  # bool: over SMA50?
        self._data_id: int | None = None

    # ------------------------------------------------------------------
    # Precomputation
    # ------------------------------------------------------------------

    def _ensure_matrix(self, data: dict[str, pd.DataFrame]) -> None:
        if self._score_matrix is not None and self._data_id == id(data):
            return

        self._data_id = id(data)

        close_series: dict[str, pd.Series] = {}
        for ticker, df in data.items():
            df = df.sort_index()
            if "close" not in df.columns or df.empty:
                continue
            s = df["close"].copy()
            s.index = pd.to_datetime(s.index).normalize()
            close_series[ticker] = s

        if not close_series:
            self._score_matrix = pd.DataFrame()
            self._trend_matrix = pd.DataFrame()
            return

        # Saml alle close i en bred matrix (dato x ticker)
        closes = pd.DataFrame(close_series)

        # Afkast over lookback-vinduer
        r1 = closes / closes.shift(self._w1) - 1
        r2 = closes / closes.shift(self._w2) - 1
        r3 = closes / closes.shift(self._w3) - 1

        # Vægtet raw score
        raw = self._wt1 * r1 + self._wt2 * r2 + self._wt3 * r3

        # Cross-sectional percentil-rang [0, 1] per dato
        # (saa vi sammenligner aktier relativt — ikke absolut afkast)
        ranked = raw.rank(axis=1, pct=True)

        # SMA50 trend-filter: True = over SMA50 = ok at koebe
        sma50 = closes.rolling(self._sma).mean()
        self._trend_matrix = closes > sma50
        self._score_matrix = ranked

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rank(
        self,
        data: dict[str, pd.DataFrame],
        as_of: date | None = None,
    ) -> pd.DataFrame:
        if not data:
            return pd.DataFrame()

        self._ensure_matrix(data)

        if self._score_matrix is None or self._score_matrix.empty:
            return pd.DataFrame()

        # Find naermeste dato i matrix (<= as_of)
        if as_of is None:
            ts = self._score_matrix.index.max()
        else:
            ts = pd.Timestamp(as_of)
            valid = self._score_matrix.index[self._score_matrix.index <= ts]
            if valid.empty:
                return pd.DataFrame()
            ts = valid[-1]

        scores = self._score_matrix.loc[ts].dropna()
        trend = self._trend_matrix.loc[ts] if self._trend_matrix is not None else None

        # Anvend SMA50-filter
        if trend is not None:
            ok = trend.reindex(scores.index).fillna(False)
            scores = scores[ok]

        if scores.empty:
            return pd.DataFrame()

        scores = scores.sort_values(ascending=False)
        result = pd.DataFrame(
            {"ticker": scores.index, "weekly_mom_score": scores.values}
        )
        result.index = range(1, len(result) + 1)
        result.index.name = "rank"
        return result.reset_index()


def build(
    rebalance_days: int = REBALANCE_DAYS,
    top_n: int = TOP_N,
    portfolio_usd: float | None = None,
) -> tuple:
    strat = WeeklyMomentumStrategy()
    if portfolio_usd is not None:
        return "Weekly Momentum", strat, rebalance_days, top_n, portfolio_usd
    return "Weekly Momentum", strat, rebalance_days, top_n

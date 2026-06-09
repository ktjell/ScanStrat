"""research/strategies/low_volatility.py

Low Volatility: ranger aktier efter lavest realiseret 30-dages volatilitet.
Historisk stærkt risikojusteret afkast — "low vol anomaly".
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datetime import date

import pandas as pd

from features.feature_engine import FeatureEngine
from features.volatility import VolatilityFeature
from ranking.normalizer import Normalizer

# ------------------------------------------------------------------
# Parametre — tilpas her
# ------------------------------------------------------------------
VOL_WINDOW: int = 30  # dage til at beregne historisk volatilitet
MIN_HISTORY: int = 60  # minimum antal dage med data


class LowVolatilityStrategy:
    """Ranger aktier fra lavest til højest realiseret volatilitet.

    Samme .rank(data, as_of) interface som Ranker og ReversalStrategy.
    """

    def __init__(self, vol_window: int = VOL_WINDOW) -> None:
        self._feat = VolatilityFeature(window=vol_window)
        self._normalizer = Normalizer(
            ascending={self._feat.name: False}  # lav vol → høj score
        )

    def rank(
        self,
        data: dict[str, pd.DataFrame],
        as_of: date | None = None,
    ) -> pd.DataFrame:
        if not data:
            return pd.DataFrame()

        engine = FeatureEngine([self._feat])
        raw = engine.compute_all(data, as_of=as_of)
        if raw.empty:
            return pd.DataFrame()

        raw = raw.dropna(subset=[self._feat.name])
        if raw.empty:
            return pd.DataFrame()

        scores = self._normalizer.normalize(raw)
        scores = scores.rename(columns={self._feat.name: "low_vol_score"})
        scores = scores.sort_values("low_vol_score", ascending=False)
        scores = scores.reset_index()  # ticker-indeks bliver kolonne
        scores.insert(0, "rank", range(1, len(scores) + 1))
        return scores


REBALANCE_DAYS: int = 30  # low vol: månedlig rebalancering


def build(
    rebalance_days: int = REBALANCE_DAYS,
) -> tuple[str, LowVolatilityStrategy, int]:
    return "Low Volatility", LowVolatilityStrategy(VOL_WINDOW), rebalance_days

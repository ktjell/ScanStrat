"""research/strategies/high52w.py

52-Week High Momentum (George & Hwang 2004):
Ranger aktier efter nærhed til 52-ugers HOJ — aktier tæt på ny top
tenderer til at fortsætte op (momentum bekræftet af nyt high).

Modsat reversal.py som ser på 52-ugers lav.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datetime import date

import pandas as pd

from features.feature_engine import FeatureEngine
from features.momentum import Momentum12M, MomentumFeature
from ranking.normalizer import Normalizer

# ------------------------------------------------------------------
# Parametre — tilpas her
# ------------------------------------------------------------------
MOMENTUM_WINDOW: int = 252  # dage til klassisk momentum-filter (12M)
MIN_MOMENTUM: float = 0.0  # kræv positiv 12M momentum (sæt til 0 for at slå fra)


class Dist52WHighFeature:
    """Nærhed til 52-ugers HOJ: close[-1] / max(high[-252:]) - 1.

    Returnerer 0.0 hvis aktien er *på* sit 52-ugers high.
    Returnerer -0.1 hvis aktien er 10% under sit 52-ugers high.
    Højere (tættere på 0) = stærkere signal.
    """

    name: str = "dist_52w_high"

    def compute(self, df: pd.DataFrame) -> float:
        close = df["close"].dropna()
        high = df["high"].dropna() if "high" in df.columns else close
        if len(close) < 2:
            return float("nan")
        window = min(252, len(high))
        high_52w = float(high.iloc[-window:].max())
        if high_52w == 0:
            return float("nan")
        return float(close.iloc[-1] / high_52w - 1)


Dist52WHigh = Dist52WHighFeature()


class High52WStrategy:
    """Ranger aktier efter nærhed til 52-ugers hoj.

    Filter: kun aktier med positiv 12M momentum (undgår faldende knive).
    Samme .rank(data, as_of) interface som Ranker og ReversalStrategy.
    """

    def __init__(
        self,
        momentum_window: int = MOMENTUM_WINDOW,
        min_momentum: float = MIN_MOMENTUM,
    ) -> None:
        self._momentum_feat = MomentumFeature(trading_days=momentum_window)
        self._high_feat = Dist52WHigh
        self._min_momentum = min_momentum
        self._normalizer = Normalizer(
            ascending={"dist_52w_high": True}  # tæt på high (høj/0) → høj score
        )

    def rank(
        self,
        data: dict[str, pd.DataFrame],
        as_of: date | None = None,
    ) -> pd.DataFrame:
        if not data:
            return pd.DataFrame()

        engine = FeatureEngine([self._momentum_feat, self._high_feat])
        raw = engine.compute_all(data, as_of=as_of)
        if raw.empty:
            return pd.DataFrame()

        raw = raw.dropna(subset=["dist_52w_high"])

        # Filter: kræv positiv momentum
        mom_col = self._momentum_feat.name
        if mom_col in raw.columns and self._min_momentum is not None:
            raw = raw[raw[mom_col] >= self._min_momentum]

        if raw.empty:
            return pd.DataFrame()

        scores = self._normalizer.normalize(raw)
        scores = scores.rename(columns={"dist_52w_high": "high52w_score"})
        scores = scores.sort_values("high52w_score", ascending=False)
        scores = scores.reset_index()  # ticker-indeks bliver kolonne
        scores.insert(0, "rank", range(1, len(scores) + 1))
        return scores


REBALANCE_DAYS: int = 30  # 52W high: månedlig rebalancering


def build(rebalance_days: int = REBALANCE_DAYS) -> tuple[str, High52WStrategy, int]:
    return "52W High Momentum", High52WStrategy(), rebalance_days

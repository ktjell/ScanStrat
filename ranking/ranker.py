from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from config.settings import RankingSettings
from features.feature_engine import FeatureEngine
from ranking.normalizer import Normalizer
from ranking.scorer import Scorer

logger = logging.getLogger(__name__)


class Ranker:
    """
    Full ranking pipeline: OHLCV data → ranked DataFrame with scores.

    Pipeline
    --------
    1. FeatureEngine.compute_all()  →  raw feature DataFrame
    2. Normalizer.normalize()       →  cross-sectional percentile scores
    3. Scorer.score()               →  weighted composite score
    4. Sort descending by score     →  ranked output

    Backtest usage
    --------------
    Pass ``as_of`` to rank the universe as of a historical date without
    leaking future data:

        result = ranker.rank(data, as_of=date(2024, 1, 1))
    """

    def __init__(
        self,
        feature_engine: FeatureEngine,
        normalizer: Normalizer,
        scorer: Scorer,
    ) -> None:
        self._feature_engine = feature_engine
        self._normalizer = normalizer
        self._scorer = scorer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rank(
        self,
        data: dict[str, pd.DataFrame],
        as_of: date | None = None,
    ) -> pd.DataFrame:
        """
        Rank all tickers in *data*.

        Returns
        -------
        pd.DataFrame
            Columns: score + all raw feature values.
            Index: integer rank (1 = best).
            A ``ticker`` column holds the ticker symbol.
        """
        if not data:
            return pd.DataFrame()

        raw = self._feature_engine.compute_all(data, as_of=as_of)
        if raw.empty:
            return pd.DataFrame()

        # Exclude tickers in an active death cross (SMA50 < SMA200).
        # death_cross == 1.0 → excluded; 0.0 or NaN → kept.
        if "death_cross" in raw.columns:
            excluded = raw["death_cross"] == 1.0
            n_excluded = excluded.sum()
            if n_excluded:
                logger.info("Excluded %d tickers (active death cross)", n_excluded)
            raw = raw[~excluded]

        if raw.empty:
            return pd.DataFrame()

        normalised = self._normalizer.normalize(raw)
        scores = self._scorer.score(normalised)

        result = raw.copy()
        result.insert(0, "score", scores.round(1))
        result = result.sort_values("score", ascending=False)
        result.index.name = "ticker"
        result = result.reset_index()
        result.insert(0, "rank", range(1, len(result) + 1))

        logger.info(
            "Ranked %d tickers%s",
            len(result),
            f" as_of={as_of}" if as_of else "",
        )
        return result

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def default(cls, settings: RankingSettings) -> Ranker:
        return cls(
            feature_engine=FeatureEngine.default(),
            normalizer=Normalizer(),
            scorer=Scorer(settings.weights),
        )

"""ReversalStrategy — mean-reversion / contrarian ranking.

Identifies stocks that:
  - have had weak recent performance (negative return over lookback)
  - are trading close to their 52-week low
  - show above-average volume (institutional accumulation signal)
  - may show a positive daily return (intraday reversal confirmation)

The reversal_score is a weighted composite of four sub-scores, each
normalised cross-sectionally to [0, 100] within the universe snapshot:

    reversal_score =
        weight_return           × negative_return_score
      + weight_volume           × volume_score
      + weight_daily_reversal   × daily_reversal_score
      + weight_proximity_to_low × proximity_to_low_score

Usage
-----
strategy = ReversalStrategy.default(settings.reversal)
result = strategy.rank(data, as_of=date(2024, 6, 1))
top10  = result.head(10)
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from config.settings import ReversalSettings
from features.feature_engine import FeatureEngine
from features.oscillators import RSI14
from features.reversal import (
    DailyReturn,
    Dist52WLow,
    Return20D,
    ReturnFeature,
    VolumeRatioFeature,
)
from ranking.normalizer import Normalizer

logger = logging.getLogger(__name__)


class ReversalStrategy:
    """
    Contrarian / mean-reversion ranking strategy.

    Parameters
    ----------
    settings:
        ``ReversalSettings`` instance controlling weights and filters.
    """

    def __init__(self, settings: ReversalSettings) -> None:
        self._s = settings
        # Build feature engine with only the features this strategy needs
        self._feature_engine = FeatureEngine(
            [
                ReturnFeature(trading_days=settings.return_lookback_days),
                VolumeRatioFeature(window=settings.volume_window),
                DailyReturn,
                Dist52WLow,
                RSI14,
            ]
        )
        return_col = f"return_{settings.return_lookback_days}d"
        volume_col = f"volume_ratio_{settings.volume_window}d"

        # Ascending=False for return and dist_52w_low: lower → better score
        self._normalizer = Normalizer(
            ascending={
                return_col: False,  # negative return → high score
                volume_col: True,  # high volume → high score
                "daily_return": True,  # positive day → high score
                "dist_52w_low": False,  # close to 52w low → high score
                "rsi_14": False,  # low RSI → oversold → high score
            }
        )
        self._return_col = return_col
        self._volume_col = volume_col

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rank(
        self,
        data: dict[str, pd.DataFrame],
        as_of: date | None = None,
    ) -> pd.DataFrame:
        """
        Rank all tickers by reversal score.

        Returns
        -------
        pd.DataFrame
            Columns: ``rank``, ``reversal_score``, raw feature values.
            Sorted descending by ``reversal_score``.
            An empty DataFrame is returned if no tickers pass filters.
        """
        if not data:
            return pd.DataFrame()

        raw = self._feature_engine.compute_all(data, as_of=as_of)
        if raw.empty:
            return pd.DataFrame()

        # --- Hard filters ---
        mask = pd.Series(True, index=raw.index)

        # 1. Return over lookback must be negative
        if self._return_col in raw.columns:
            mask &= raw[self._return_col] < 0

        # 2. RSI below threshold (oversold zone)
        if "rsi_14" in raw.columns:
            mask &= raw["rsi_14"] < self._s.rsi_threshold

        # 3. Volume ratio above threshold
        if self._volume_col in raw.columns:
            mask &= raw[self._volume_col] >= self._s.volume_ratio_threshold

        # 4. Must be near 52-week low
        if self._s.max_dist_52w_low is not None and "dist_52w_low" in raw.columns:
            mask &= raw["dist_52w_low"] <= self._s.max_dist_52w_low

        # 5. Optional: last day must be positive (confirmed reversal candle)
        if self._s.require_positive_day and "daily_return" in raw.columns:
            mask &= raw["daily_return"] > 0

        filtered = raw[mask]
        n_excluded = (~mask).sum()
        if n_excluded:
            logger.info(
                "ReversalStrategy: excluded %d tickers (hard filters), %d remain",
                n_excluded,
                len(filtered),
            )

        if filtered.empty:
            logger.warning(
                "ReversalStrategy: no tickers passed hard filters%s",
                f" as_of={as_of}" if as_of else "",
            )
            return pd.DataFrame()

        normalised = self._normalizer.normalize(filtered)
        scores = self._compute_score(normalised)

        result = filtered.copy()
        result.insert(0, "reversal_score", scores.round(1))
        result = result.sort_values("reversal_score", ascending=False)
        result.index.name = "ticker"
        result = result.reset_index()
        result.insert(0, "rank", range(1, len(result) + 1))

        logger.info(
            "ReversalStrategy ranked %d tickers%s",
            len(result),
            f" as_of={as_of}" if as_of else "",
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_score(self, normalised: pd.DataFrame) -> pd.Series:
        s = self._s
        total_weight = (
            s.weight_return
            + s.weight_volume
            + s.weight_daily_reversal
            + s.weight_proximity_to_low
        )

        score = pd.Series(0.0, index=normalised.index)
        _NAN_FILL = 50.0

        def _add(col: str, weight: float) -> None:
            if col in normalised.columns:
                score.__iadd__(
                    (weight / total_weight) * normalised[col].fillna(_NAN_FILL)
                )

        _add(self._return_col, s.weight_return)
        _add(self._volume_col, s.weight_volume)
        _add("daily_return", s.weight_daily_reversal)
        _add("dist_52w_low", s.weight_proximity_to_low)

        return score

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def default(cls, settings: ReversalSettings | None = None) -> ReversalStrategy:
        from config.settings import ReversalSettings as RS

        return cls(settings or RS())

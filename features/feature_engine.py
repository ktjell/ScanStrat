from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from features.base import Feature
from features.momentum import Momentum3M, Momentum6M, Momentum12M
from features.oscillators import RSI14
from features.trend import Dist52WHigh, SMA50, SMA200, DeathCross
from features.volatility import Volatility30D

logger = logging.getLogger(__name__)

_DEFAULT_FEATURES: list[Feature] = [
    Momentum3M,
    Momentum6M,
    Momentum12M,
    SMA50,
    SMA200,
    Dist52WHigh,
    DeathCross,
    RSI14,
    Volatility30D,
]


class FeatureEngine:
    """
    Applies a configurable set of features to OHLCV data.

    Backtest usage
    --------------
    To compute features as of a historical date, slice the DataFrame first:

        as_of_df = full_df.loc[:str(as_of_date)]
        row = engine.compute_row("AAPL", as_of_df)

    This design means the same engine can be used for both live analysis
    and historical backtests without any code changes.
    """

    def __init__(self, features: list[Feature]) -> None:
        self._features = features

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_row(self, ticker: str, df: pd.DataFrame) -> dict[str, float | str]:
        """
        Compute all features for a single ticker.

        Returns a dict with ``ticker`` as the first key, followed by one
        float value per feature.  NaN is returned for features with
        insufficient data.
        """
        result: dict[str, float | str] = {"ticker": ticker}
        for feature in self._features:
            value = feature.compute(df)
            if pd.isna(value):
                logger.debug(
                    "Feature '%s' returned NaN for '%s' (%d rows)",
                    feature.name,
                    ticker,
                    len(df),
                )
            result[feature.name] = value
        return result

    def compute_all(
        self,
        data: dict[str, pd.DataFrame],
        as_of: date | None = None,
    ) -> pd.DataFrame:
        """
        Compute features for all tickers in *data*.

        Parameters
        ----------
        data   : dict mapping ticker → OHLCV DataFrame
        as_of  : optional date; if provided, each DataFrame is sliced
                 to ``df.loc[:str(as_of)]`` before feature computation
                 (backtest mode).

        Returns
        -------
        pd.DataFrame
            One row per ticker, one column per feature, indexed by ticker.
        """
        rows = []
        for ticker, df in data.items():
            sliced = df.loc[: str(as_of)] if as_of is not None else df
            rows.append(self.compute_row(ticker, sliced))

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows).set_index("ticker")

    @property
    def feature_names(self) -> list[str]:
        return [f.name for f in self._features]

    @classmethod
    def default(cls) -> FeatureEngine:
        """Return an engine with the standard feature set."""
        return cls(list(_DEFAULT_FEATURES))

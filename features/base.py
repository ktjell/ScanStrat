from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class Feature(Protocol):
    """
    Structural interface for all technical features.

    A Feature computes a single float value from an OHLCV DataFrame.
    `compute` always operates on the *last* row of whatever data is passed,
    so backtesting is trivial: slice `df` to the as_of date, then call `compute`.

    Example (backtest usage)
    ------------------------
    as_of_df = full_df.loc[:str(as_of_date)]
    value = Momentum6M.compute(as_of_df)
    """

    @property
    def name(self) -> str:
        """Unique, stable identifier used as the column name in output DataFrames."""
        ...

    def compute(self, df: pd.DataFrame) -> float:
        """
        Compute the feature value.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame conforming to the canonical schema
            (DatetimeIndex, columns: open/high/low/close/volume).
            May be a full history or a slice up to a given date.

        Returns
        -------
        float
            The computed value, or ``float("nan")`` if there is
            insufficient data.
        """
        ...

from __future__ import annotations

from datetime import date
from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class PriceDataLoader(Protocol):
    """
    Structural interface for all price-data sources.

    Any class that implements `fetch` and `fetch_batch` with the
    correct signatures is automatically a valid PriceDataLoader —
    no inheritance required (duck typing via Protocol).

    Returned DataFrames must conform to the canonical schema
    defined in data.models.price_data:
        Index   : DatetimeIndex (UTC-naive), name="date"
        Columns : open, high, low, close, volume
    """

    def fetch(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        """Fetch OHLCV data for a single ticker."""
        ...

    def fetch_batch(
        self,
        tickers: list[str],
        start: date,
        end: date,
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for multiple tickers.

        Tickers for which no data is available are omitted from the result.
        """
        ...

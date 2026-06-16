from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

# Brug en projekt-lokal mappe til yfinance's timezone-cache i stedet for den
# globale ~/.cache/py-yfinance som korrupterer ved samtidige skrivninger.
# Den lokale mappe overlever mellem kørsler (så tz kun slås op én gang per ticker)
# men er isoleret fra andre processer.
_YF_TZ_CACHE = Path(__file__).parent.parent.parent / "data" / "cache" / ".yf_tz"
_YF_TZ_CACHE.mkdir(parents=True, exist_ok=True)
yf.set_tz_cache_location(str(_YF_TZ_CACHE))  # type: ignore[attr-defined]

from data.models.price_data import validate_price_df

logger = logging.getLogger(__name__)

# Map raw yfinance column names -> canonical schema column names
_COLUMN_MAP: dict[str, str] = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
}


class YFinanceLoader:
    """
    PriceDataLoader backed by yfinance.

    Uses auto_adjust=True so `close` is already split- and
    dividend-adjusted; no separate `adj_close` column is present.

    To swap in a different data source (e.g. Saxo OpenAPI), implement
    the same `fetch` / `fetch_batch` signatures — no other changes are
    needed thanks to the PriceDataLoader Protocol.
    """

    def __init__(self, timeout: int = 30) -> None:
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        logger.debug("Fetching %s [%s → %s] via yfinance", ticker, start, end)
        raw: pd.DataFrame = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            timeout=self._timeout,
        )
        if raw.empty:
            raise ValueError(f"No data returned for ticker '{ticker}'")
        return self._normalize(raw, ticker)

    def fetch_batch(
        self,
        tickers: list[str],
        start: date,
        end: date,
    ) -> dict[str, pd.DataFrame]:
        if not tickers:
            return {}

        logger.debug(
            "Batch-fetching %d tickers [%s → %s] via yfinance",
            len(tickers),
            start,
            end,
        )

        raw: pd.DataFrame = yf.download(
            tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            timeout=self._timeout,
            group_by="ticker",
        )

        # Single-ticker download returns a flat DataFrame
        if len(tickers) == 1:
            ticker = tickers[0]
            if raw.empty:
                logger.warning("No data for ticker '%s'", ticker)
                return {}
            return {ticker: self._normalize(raw, ticker)}

        # Multi-ticker download: columns are a (ticker, price) MultiIndex
        result: dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            try:
                ticker_df: pd.DataFrame = raw[ticker]
            except KeyError:
                logger.warning("Ticker '%s' not found in batch result", ticker)
                continue
            if ticker_df.dropna(how="all").empty:
                logger.warning("No data for ticker '%s'", ticker)
                continue
            result[ticker] = self._normalize(ticker_df, ticker)

        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        # Flatten MultiIndex columns — yfinance producerer to formater afhaengig
        # af group_by og antal tickers:
        #   Format A (ingen group_by, enkelt ticker): ('Close', 'VWCE.DE')
        #   Format B (group_by='ticker', enkelt ticker): ('VWCE.DE', 'Close')
        # Vi finder det niveau der indeholder kendte prisnavne og beholder det.
        if isinstance(df.columns, pd.MultiIndex):
            price_names = set(_COLUMN_MAP.keys())
            for level in range(df.columns.nlevels):
                if set(df.columns.get_level_values(level)) & price_names:
                    df.columns = df.columns.get_level_values(level)
                    break

        df = df.rename(columns=_COLUMN_MAP)

        # Keep only the columns we care about; ignore Dividends, Stock Splits, etc.
        available = [c for c in _COLUMN_MAP.values() if c in df.columns]
        df = df[available]

        # Ensure the index is a tz-naive DatetimeIndex
        df.index = pd.to_datetime(df.index).tz_localize(None)

        return validate_price_df(df, ticker)

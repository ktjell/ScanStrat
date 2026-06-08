from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from config.settings import Settings
from data.cache.bad_ticker_cache import BadTickerCache
from data.cache.cache_manager import CacheManager
from data.loaders.base import PriceDataLoader

logger = logging.getLogger(__name__)


class DataService:
    """
    Orchestrates data fetching and caching.

    The concrete loader is injected at construction time, so the
    DataService itself has zero dependency on yfinance or any other
    provider — swap `YFinanceLoader` for `SaxoLoader` and nothing else
    needs to change.

    Usage
    -----
    service = DataService(loader=YFinanceLoader(), cache=CacheManager(settings.cache), settings=settings)
    df = service.get("AAPL", start=date(2020, 1, 1), end=date(2024, 12, 31))
    """

    def __init__(
        self,
        loader: PriceDataLoader,
        cache: CacheManager,
        settings: Settings,
    ) -> None:
        self._loader = loader
        self._cache = cache
        self._settings = settings
        self._bad = BadTickerCache(Path(settings.cache.cache_dir))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(
        self,
        ticker: str,
        start: date | None = None,
        end: date | None = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Return OHLCV data for *ticker*.

        Data is served from cache when fresh; otherwise fetched from
        the underlying loader and then persisted to cache.
        """
        start, end = self._resolve_date_range(start, end)

        if not force_refresh and self._cache_covers(ticker, start, end):
            logger.info("Cache hit  '%s'", ticker)
            df = self._cache.load(ticker)
            return df.loc[str(start) : str(end)]

        logger.info("Fetching   '%s' from source", ticker)
        df = self._loader.fetch(ticker, start, end)
        self._cache.save(ticker, df)
        return df

    def get_batch(
        self,
        tickers: list[str],
        start: date | None = None,
        end: date | None = None,
        force_refresh: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """
        Return OHLCV data for multiple tickers.

        Fresh tickers are served from cache; stale tickers are fetched
        in a single batch call to minimise round trips.
        Known-bad tickers (delisted, wrong symbol) are skipped silently.
        """
        start, end = self._resolve_date_range(start, end)

        # Filter known-bad tickers before any network activity
        good_tickers, skipped = self._bad.filter_good(tickers)
        if skipped:
            logger.debug("Skipping %d known-bad tickers", len(skipped))

        cached: dict[str, pd.DataFrame] = {}
        to_fetch: list[str] = []

        for ticker in good_tickers:
            if not force_refresh and self._cache_covers(ticker, start, end):
                try:
                    df = self._cache.load(ticker)
                    cached[ticker] = df.loc[str(start) : str(end)]
                    continue
                except Exception:
                    pass  # fall through to fetch
            to_fetch.append(ticker)

        if to_fetch:
            logger.info("Fetching %d ticker(s) from source", len(to_fetch))
            fetched = self._loader.fetch_batch(to_fetch, start, end)

            # Mark tickers that yfinance returned nothing for as bad
            failed = [t for t in to_fetch if t not in fetched]
            if failed:
                logger.info(
                    "Marking %d ticker(s) as bad (no data returned): %s",
                    len(failed),
                    failed[:10],
                )
                self._bad.mark_bad_batch(failed)

            for ticker, df in fetched.items():
                self._cache.save(ticker, df)
                cached[ticker] = df.loc[str(start) : str(end)]

        return cached

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def count_cached(self, tickers: list[str], start: date, end: date) -> int:
        """Returner antal tickers der allerede er dækket af disk-cachen."""
        good, _ = self._bad.filter_good(tickers)
        return sum(1 for t in good if self._cache_covers(t, start, end))

    def _cache_covers(self, ticker: str, start: date, end: date) -> bool:
        """
        Return True if the cache is fresh AND covers the requested date range.

        A 7-day tolerance on both ends accounts for weekends and public holidays
        (the first/last trading day may not fall exactly on start/end).
        """
        if not self._cache.is_fresh(ticker):
            return False
        try:
            df = self._cache.load(ticker)
        except FileNotFoundError:
            return False
        if df.empty:
            return False
        tolerance = timedelta(days=7)
        cache_start = df.index[0].date()
        cache_end = df.index[-1].date()
        return cache_start <= start + tolerance and cache_end >= end - tolerance

    def _resolve_date_range(
        self,
        start: date | None,
        end: date | None,
    ) -> tuple[date, date]:
        end = end or date.today()
        start = start or end - timedelta(
            days=365 * self._settings.data.default_period_years
        )
        return start, end

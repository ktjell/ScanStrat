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

        if not force_refresh and self._cache.is_fresh(ticker):
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

        Friske tickers (cache dækker til seneste handelsdag) serveres fra disk.
        Bagud-daterede tickers: hent kun de manglende dage og append til cache.
        Ukendte tickers: hent al historik og gem.
        Known-bad tickers (delisted, wrong symbol) er skippet.
        """
        start, end = self._resolve_date_range(start, end)

        # Filter known-bad tickers before any network activity
        good_tickers, skipped = self._bad.filter_good(tickers)
        if skipped:
            logger.debug("Skipping %d known-bad tickers", len(skipped))

        cached: dict[str, pd.DataFrame] = {}
        to_fetch_full: list[str] = []  # ingen cache — hent al historik
        to_fetch_update: dict[str, date] = {}  # cache bagud — hent fra denne dato

        for ticker in good_tickers:
            if not force_refresh and self._cache.is_fresh(ticker):
                # Cache er up-to-date — brug den direkte
                try:
                    df = self._cache.load(ticker)
                    if not df.empty and df.index[0].date() <= start + timedelta(days=7):
                        cached[ticker] = df.loc[str(start) : str(end)]
                        continue
                except Exception:
                    pass
            # Tjek om vi har gammel cache der blot mangler nyeste dage
            last = self._cache.last_date(ticker)
            if last is not None and last >= start:
                # Vi har historik — hent kun fra dagen efter seneste kendte dato
                to_fetch_update[ticker] = last + timedelta(days=1)
            else:
                to_fetch_full.append(ticker)

        # Hent tickers uden cache (fuld historik)
        if to_fetch_full:
            logger.info(
                "Fetching %d ticker(s) from source (full history)", len(to_fetch_full)
            )
            fetched = self._loader.fetch_batch(to_fetch_full, start, end)
            failed = [t for t in to_fetch_full if t not in fetched]
            if failed:
                # Prøv at bruge eksisterende cache-data som fallback inden vi opgiver
                truly_bad: list[str] = []
                for ticker in failed:
                    try:
                        df = self._cache.load(ticker)
                        if not df.empty:
                            cached[ticker] = df.loc[str(start) : str(end)]
                            logger.debug(
                                "Cache-fallback for '%s' (fetch fejlede, bruger eksisterende data)",
                                ticker,
                            )
                        else:
                            truly_bad.append(ticker)
                    except Exception:
                        truly_bad.append(ticker)
                if truly_bad:
                    logger.info(
                        "Marking %d ticker(s) as bad (ingen cache-fallback): %s",
                        len(truly_bad),
                        truly_bad[:10],
                    )
                    self._bad.mark_bad_batch(truly_bad)
            for ticker, df in fetched.items():
                self._cache.save(ticker, df)
                cached[ticker] = df.loc[str(start) : str(end)]

        # Hent tickers med gammel cache (kun manglende dage)
        if to_fetch_update:
            # Gruppér alle update-tickers i ét batch-kald med fælles start = tidligste manglende dato
            earliest_update = min(to_fetch_update.values())
            logger.info(
                "Updating %d ticker(s) from %s (appending missing days)",
                len(to_fetch_update),
                earliest_update,
            )
            fetched_update = self._loader.fetch_batch(
                list(to_fetch_update.keys()), earliest_update, end
            )
            failed_update = [t for t in to_fetch_update if t not in fetched_update]
            if failed_update:
                logger.debug(
                    "No new data for %d ticker(s) (expected for recent cache)",
                    len(failed_update),
                )
            for ticker, new_df in fetched_update.items():
                # Append kun rækker efter seneste kendte dato
                cutoff = to_fetch_update[ticker]
                new_rows = (
                    new_df[new_df.index.date >= cutoff] if not new_df.empty else new_df
                )
                if not new_rows.empty:
                    self._cache.append(ticker, new_rows)
                # Load den nu-opdaterede cache
                try:
                    df = self._cache.load(ticker)
                    cached[ticker] = df.loc[str(start) : str(end)]
                except Exception:
                    pass
            # Tickers der ikke fik nye data — brug eksisterende cache som-er
            for ticker in to_fetch_update:
                if ticker not in cached:
                    try:
                        df = self._cache.load(ticker)
                        cached[ticker] = df.loc[str(start) : str(end)]
                    except Exception:
                        pass

        return cached

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def count_cached(self, tickers: list[str], start: date, end: date) -> int:
        """Returner antal tickers der allerede er dækket af disk-cachen."""
        good, _ = self._bad.filter_good(tickers)
        return sum(1 for t in good if self._cache.is_fresh(t))

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

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from config.settings import CacheSettings
from data.models.price_data import validate_price_df

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Parquet-based local cache for price data.

    Each ticker is stored as a single .parquet file under `cache_dir`.
    Staleness is determined by the file's mtime relative to `max_age_hours`.

    Ticker names containing path-unsafe characters (e.g. BRK/B) are
    sanitised before being used as file names.
    """

    def __init__(self, settings: CacheSettings) -> None:
        self._dir = Path(settings.cache_dir)
        self._max_age = timedelta(hours=settings.max_age_hours)
        self._dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def exists(self, ticker: str) -> bool:
        """Return True if a cache file exists for *ticker*."""
        return self._path(ticker).exists()

    def is_fresh(self, ticker: str) -> bool:
        """
        Return True if the cache file exists **and** was written
        within the configured max_age window.
        """
        path = self._path(ticker)
        if not path.exists():
            return False
        age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
        return age < self._max_age

    def load(self, ticker: str) -> pd.DataFrame:
        """Load and return the cached DataFrame for *ticker*."""
        path = self._path(ticker)
        if not path.exists():
            raise FileNotFoundError(f"No cache entry for ticker '{ticker}'")
        logger.debug("Cache load  '%s'", ticker)
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        return df

    def save(self, ticker: str, df: pd.DataFrame) -> None:
        """Validate *df* against the canonical schema and persist it."""
        df = validate_price_df(df, ticker)
        path = self._path(ticker)
        logger.debug("Cache save  '%s' (%d rows)", ticker, len(df))
        df.to_parquet(path)

    def invalidate(self, ticker: str) -> None:
        """Delete the cache file for *ticker*, if it exists."""
        path = self._path(ticker)
        if path.exists():
            path.unlink()
            logger.debug("Cache invalidated '%s'", ticker)

    def clear_all(self) -> None:
        """Delete every .parquet file in the cache directory."""
        for path in self._dir.glob("*.parquet"):
            path.unlink()
        logger.info("Cache cleared (%s)", self._dir)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _path(self, ticker: str) -> Path:
        safe = ticker.replace("/", "_").replace("\\", "_").replace(":", "_")
        return self._dir / f"{safe}.parquet"

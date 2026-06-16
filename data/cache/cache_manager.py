from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from config.settings import CacheSettings
from data.models.price_data import validate_price_df

logger = logging.getLogger(__name__)


def _last_trading_day() -> date:
    """
    Seneste AFSLUTTEDE handelsdato.

    US-markedet lukker kl. 22:00 CET (16:00 ET). Hvis det er inden
    lukketid i dag, er gårsdagens (eller fredagens) data den seneste
    komplette dag — ellers ville vi forsøge at opdatere til i dag og
    yfinance ville fejle med 'possibly delisted'.
    """
    # US market close = 22:00 CET = 20:00 UTC
    now_utc = datetime.now(tz=timezone.utc)
    us_close_utc = now_utc.replace(hour=20, minute=30, second=0, microsecond=0)

    d = date.today()
    # Hvis markedet ikke er lukket endnu i dag, gå en dag tilbage
    if now_utc < us_close_utc:
        d -= timedelta(days=1)
    # Gå tilbage til fredag hvis lørdag/søndag
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


class CacheManager:
    """
    Parquet-based local cache for price data.

    Cache udløber ALDRIG baseret på alder — historiske data ændrer sig ikke.
    I stedet anses cachen som "frisk" hvis den indeholder data frem til
    seneste handelsdag (dvs. i dag hvis hverdagsdag, ellers fredag).
    DataService appender kun de manglende dage når cachen er bagud.
    """

    def __init__(self, settings: CacheSettings) -> None:
        self._dir = Path(settings.cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def exists(self, ticker: str) -> bool:
        """Return True if a cache file exists for *ticker*."""
        return self._path(ticker).exists()

    def is_fresh(self, ticker: str) -> bool:
        """
        Return True hvis cache-filen eksisterer OG indeholder data
        frem til seneste handelsdato.
        Historiske data ændrer sig ikke — vi tjekker kun om nyeste dag mangler.
        """
        last = self.last_date(ticker)
        if last is None:
            return False
        return last >= _last_trading_day()

    def load(self, ticker: str) -> pd.DataFrame:
        """Load and return the cached DataFrame for *ticker*."""
        path = self._path(ticker)
        if not path.exists():
            raise FileNotFoundError(f"No cache entry for ticker '{ticker}'")
        logger.debug("Cache load  '%s'", ticker)
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        return df

    def last_date(self, ticker: str) -> date | None:
        """Returnerer seneste dato i cachen for *ticker*, eller None."""
        path = self._path(ticker)
        if not path.exists():
            return None
        try:
            # Læs kun index ved at hente én kolonne — undgår df.empty-fælden
            # når columns=[] giver en DataFrame med rækker men ingen kolonner
            df = pd.read_parquet(path)
            df.index = pd.to_datetime(df.index)
            if len(df.index) == 0:
                return None
            return df.index.max().date()
        except Exception:
            return None

    def save(self, ticker: str, df: pd.DataFrame) -> None:
        """Validate *df* against the canonical schema and persist it."""
        df = validate_price_df(df, ticker)
        path = self._path(ticker)
        logger.debug("Cache save  '%s' (%d rows)", ticker, len(df))
        df.to_parquet(path)

    def append(self, ticker: str, new_df: pd.DataFrame) -> None:
        """
        Append nye rækker til eksisterende cache.
        Duplikater (samme dato) overskrives med nye værdier.
        """
        if new_df.empty:
            return
        path = self._path(ticker)
        if path.exists():
            existing = self.load(ticker)
            combined = pd.concat([existing, new_df])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined = combined.sort_index()
        else:
            combined = new_df
        self.save(ticker, combined)
        logger.debug("Cache append '%s' (+%d rows)", ticker, len(new_df))

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

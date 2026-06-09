"""data/cache/bad_ticker_cache.py

Persistent cache af tickers der konsekvent fejler (aflistede, forkert symbol osv.).
Gemmes som JSON i samme mappe som parquet-filerne.

Tickers forbliver "bad" i BAD_TTL_DAYS dage, herefter forsoeges de igen
(i tilfaelde af midlertidige problemer).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

BAD_TTL_DAYS: int = 30
_FILENAME: str = "bad_tickers.json"


class BadTickerCache:
    """Fil-baseret cache over kendte-fejlede tickers med TTL."""

    def __init__(self, cache_dir: Path) -> None:
        self._path = cache_dir / _FILENAME
        self._data: dict[str, str] = {}  # ticker -> ISO-timestamp
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_bad(self, ticker: str) -> bool:
        """Returner True hvis ticker er markeret bad og TTL ikke er udloebet."""
        ts_str = self._data.get(ticker)
        if ts_str is None:
            return False
        try:
            marked_at = datetime.fromisoformat(ts_str)
        except ValueError:
            return False
        if datetime.now(tz=timezone.utc) - marked_at > timedelta(days=BAD_TTL_DAYS):
            # TTL udloebet — fjern fra listen og proev igen
            del self._data[ticker]
            self._save()
            return False
        return True

    def mark_bad(self, ticker: str) -> None:
        """Marker ticker som bad med nuvaerende tidsstempel."""
        self._data[ticker] = datetime.now(tz=timezone.utc).isoformat()
        self._save()

    def mark_bad_batch(self, tickers: list[str]) -> None:
        """Marker en liste tickers som bad paa een gang (enkelt disk-skrivning)."""
        if not tickers:
            return
        now = datetime.now(tz=timezone.utc).isoformat()
        for t in tickers:
            self._data[t] = now
        self._save()

    def filter_good(self, tickers: list[str]) -> tuple[list[str], list[str]]:
        """Returner (gode, bad) opdelt fra input-listen."""
        good: list[str] = []
        bad: list[str] = []
        for t in tickers:
            (bad if self.is_bad(t) else good).append(t)
        return good, bad

    def count(self) -> int:
        return len(self._data)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning(
                    "Could not read bad ticker cache (%s) — starting fresh", exc
                )
                self._data = {}

    def _save(self) -> None:
        try:
            self._path.write_text(
                json.dumps(self._data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Could not save bad ticker cache: %s", exc)

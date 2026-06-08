"""data/universes/screener.py

Dynamisk univers via Yahoo Finance Screener API.

I stedet for at vedligeholde hardkodede ticker-lister hentes aktier live
baseret paa filtre (markedsvaerdi, volumen, boers, pris).

Resultaterne caches lokalt i BAD_TICKER_TTL timer saa vi ikke kalder
Yahoo Finance ved hvert genstart.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Hvor laenge univers-cachen er gyldig (timer)
_CACHE_TTL_HOURS: int = 24
_CACHE_DIR: Path = Path("data/cache")
_CACHE_FILE: Path = _CACHE_DIR / "universe_screener.json"


# ---------------------------------------------------------------------------
# Offentlig API
# ---------------------------------------------------------------------------


def get_screener_universe(
    region: str = "us",
    min_market_cap: float = 300e6,
    max_market_cap: float = 50e9,
    min_price: float = 2.0,
    min_avg_volume: float = 300_000,
    max_results: int = 500,
    force_refresh: bool = False,
) -> list[str]:
    """Hent likvide aktier via Yahoo Finance Screener.

    Parameters
    ----------
    region:
        Yahoo Finance region-kode: ``"us"``, ``"europe"``, ``"de"``, ``"fr"``,
        ``"gb"``, ``"dk"``, ``"se"``, ``"no"``, ``"fi"``, ``"nl"``, ``"es"``, ``"it"``
    min_market_cap:
        Mindste markedsvaerdi i USD (default 300M = mellemstore small caps)
    max_market_cap:
        Stoerste markedsvaerdi i USD (default 50B = undgaar megacaps)
    min_price:
        Mindste aktiekurs i USD/lokal valuta (filtrer penny stocks)
    min_avg_volume:
        Mindste gennemsnitlig dagsomsaetning i antal aktier
    max_results:
        Maks antal tickers at returnere (Yahoo max er ca. 250 pr. kald)
    force_refresh:
        Ignorer cache og hent frisk data

    Returns
    -------
    list[str]
        Sorteret liste af ticker-symboler
    """
    cache_key = f"{region}_{int(min_market_cap / 1e6)}M_{int(max_market_cap / 1e6)}M_{max_results}"

    if not force_refresh:
        cached = _load_cache(cache_key)
        if cached is not None:
            logger.info("Screener cache hit: %d tickers for %s", len(cached), region)
            return cached

    tickers = _fetch_from_yahoo(
        region, min_market_cap, max_market_cap, min_price, min_avg_volume, max_results
    )
    if tickers:
        _save_cache(cache_key, tickers)
    return tickers


def get_sp500_screener(fallback: list[str] | None = None) -> list[str]:
    """US large caps (>10B markedsvaerdi) som approksimation for S&P 500."""
    result = get_screener_universe(
        region="us",
        min_market_cap=10e9,
        max_market_cap=5_000e9,
        min_price=5.0,
        min_avg_volume=500_000,
        max_results=600,
    )
    if result:
        return result
    logger.warning("Screener fejlede for S&P 500 — bruger fallback")
    return fallback or []


def get_nasdaq100_screener(fallback: list[str] | None = None) -> list[str]:
    """NASDAQ-listede large/mega caps som approksimation for NASDAQ-100."""
    result = get_screener_universe(
        region="us",
        min_market_cap=10e9,
        max_market_cap=5_000e9,
        min_price=5.0,
        min_avg_volume=1_000_000,
        max_results=150,
    )
    if result:
        return result
    logger.warning("Screener fejlede for NASDAQ-100 — bruger fallback")
    return fallback or []


def get_us_smallcap_screener(max_results: int = 400) -> list[str]:
    """US small/mid caps: 300M - 10B markedsvaerdi, god likviditet."""
    return get_screener_universe(
        region="us",
        min_market_cap=300e6,
        max_market_cap=10e9,
        min_price=2.0,
        min_avg_volume=500_000,
        max_results=max_results,
    )


def get_eu_screener(max_results: int = 300) -> list[str]:
    """Europaeiske aktier: 200M - 20B markedsvaerdi."""
    tickers: list[str] = []
    seen: set[str] = set()
    per_region = max(max_results // 5, 60)
    for region in ["de", "fr", "gb", "se", "dk", "no", "nl", "fi", "es", "it"]:
        batch = get_screener_universe(
            region=region,
            min_market_cap=200e6,
            max_market_cap=20e9,
            min_price=0.5,
            min_avg_volume=50_000,
            max_results=per_region,
        )
        for t in batch:
            if t not in seen:
                seen.add(t)
                tickers.append(t)
    return tickers[:max_results]


# ---------------------------------------------------------------------------
# Intern implementering
# ---------------------------------------------------------------------------


def _fetch_from_yahoo(
    region: str,
    min_market_cap: float,
    max_market_cap: float,
    min_price: float,
    min_avg_volume: float,
    max_results: int,
) -> list[str]:
    try:
        from yfinance.screener.screener import EquityQuery, screen
    except ImportError:
        logger.error("yfinance screener API ikke tilgaengelig — opdater yfinance")
        return []

    try:
        filters = [
            EquityQuery("gt", ["intradaymarketcap", min_market_cap]),
            EquityQuery("lt", ["intradaymarketcap", max_market_cap]),
            EquityQuery("gt", ["intradayprice", min_price]),
            EquityQuery("gt", ["avgdailyvol3m", min_avg_volume]),
            EquityQuery("eq", ["region", region]),
        ]
        query = EquityQuery("and", filters)

        tickers: list[str] = []
        batch_size = min(250, max_results)
        offset = 0

        while len(tickers) < max_results:
            result = screen(
                query,
                sortField="avgdailyvol3m",
                sortAsc=False,
                size=min(batch_size, max_results - len(tickers)),
                offset=offset,
            )
            quotes = result.get("quotes", [])
            if not quotes:
                break
            for q in quotes:
                sym = q.get("symbol", "")
                if sym:
                    tickers.append(sym)
            offset += len(quotes)
            if offset >= result.get("total", 0):
                break

        logger.info("Screener: hentede %d tickers for region=%s", len(tickers), region)
        return tickers

    except Exception as exc:
        logger.warning("Yahoo screener fejlede for region=%s: %s", region, exc)
        return []


def _load_cache(key: str) -> list[str] | None:
    if not _CACHE_FILE.exists():
        return None
    try:
        data: dict = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
        entry = data.get(key)
        if not entry:
            return None
        fetched_at = datetime.fromisoformat(entry["fetched_at"])
        if datetime.now(tz=timezone.utc) - fetched_at > timedelta(
            hours=_CACHE_TTL_HOURS
        ):
            return None
        return entry["tickers"]
    except Exception:
        return None


def _save_cache(key: str, tickers: list[str]) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        data: dict = {}
        if _CACHE_FILE.exists():
            try:
                data = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
            except Exception:
                data = {}
        data[key] = {
            "fetched_at": datetime.now(tz=timezone.utc).isoformat(),
            "tickers": tickers,
        }
        _CACHE_FILE.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception as exc:
        logger.warning("Kunne ikke gemme screener cache: %s", exc)

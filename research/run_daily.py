"""research/run_daily.py

Nat-job: opdater data-cache med dagens priser for hele universet.
Kores dagligt (fx kl. 06:30) via Synology Task Scheduler.

Det eneste dette script goer:
  1. Byg univers-liste (samme som app og backtest)
  2. Kald DataService.get_batch() — den henter KUN nye priser for
     tickers der allerede er i cache, og fuldt for nye tickers.
  3. Faerdigt — dashboard kan nu kore Daily Breakout screener
     med friske priser uden at hente noget.

Koer med:
    uv run python research/run_daily.py
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from engine.data import (
    CacheManager,
    DataService,
    YFinanceLoader,
    get_eu_smallcap_tickers,
    get_eurostoxx50_tickers,
    get_nasdaq100_tickers,
    get_smallcap_tickers,
    get_sp500_tickers,
)
from engine.config import Settings


def _build_universe() -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for t in (
        get_sp500_tickers()
        + get_nasdaq100_tickers()
        + get_smallcap_tickers()
        + get_eurostoxx50_tickers()
        + get_eu_smallcap_tickers()
    ):
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def main() -> None:
    settings = Settings.default()
    service = DataService(YFinanceLoader(), CacheManager(settings.cache), settings)

    universe = _build_universe()
    end = date.today()
    start = end - timedelta(days=400)  # daekker 252 handelsdage + buffer

    n_cached = service.count_cached(universe, start, end)
    n_fresh = len(universe) - n_cached
    print(f"Univers: {len(universe)} aktier")
    print(f"Cache: {n_cached} up-to-date, {n_fresh} skal opdateres")

    if n_fresh == 0:
        print("Intet at hente - cache er allerede opdateret.")
        return

    print("Opdaterer...")
    data = service.get_batch(universe, start, end)
    print(f"Faerdig. {len(data)} tickers klar til screener.")


if __name__ == "__main__":
    main()

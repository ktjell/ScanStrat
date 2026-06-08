"""
ScanStrat – Phase 1 entry point.

Demonstrates the full data layer: Settings → YFinanceLoader → CacheManager → DataService.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

from config.settings import Settings
from data.cache.cache_manager import CacheManager
from data.data_service import DataService
from data.loaders.yfinance_loader import YFinanceLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

CONFIG_PATH = Path("config/config.yaml")


def build_service(settings: Settings) -> DataService:
    loader = YFinanceLoader(timeout=settings.data.yfinance_timeout)
    cache = CacheManager(settings.cache)
    return DataService(loader=loader, cache=cache, settings=settings)


def main() -> None:
    settings = (
        Settings.from_yaml(CONFIG_PATH) if CONFIG_PATH.exists() else Settings.default()
    )
    service = build_service(settings)

    tickers = ["AAPL", "MSFT", "NVDA"]
    end = date.today()
    start = end - timedelta(days=365)

    logger.info("Fetching data for: %s", tickers)
    results = service.get_batch(tickers, start=start, end=end)

    for ticker, df in results.items():
        logger.info(
            "%-6s  %d rows  %s → %s",
            ticker,
            len(df),
            df.index[0].date(),
            df.index[-1].date(),
        )


if __name__ == "__main__":
    main()

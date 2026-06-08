"""engine.data — data fetching, caching and universes."""

from data.cache.cache_manager import CacheManager
from data.data_service import DataService
from data.loaders.yfinance_loader import YFinanceLoader
from data.models.price_data import validate_price_df
from data.universes.eu_smallcap import get_eu_smallcap_tickers
from data.universes.eurostoxx50 import get_eurostoxx50_tickers
from data.universes.nasdaq100 import get_nasdaq100_tickers
from data.universes.smallcap import get_smallcap_tickers
from data.universes.screener import (
    get_eu_screener,
    get_nasdaq100_screener,
    get_screener_universe,
    get_sp500_screener,
    get_us_smallcap_screener,
)
from data.universes.sp500 import get_sp500_tickers

__all__ = [
    "CacheManager",
    "DataService",
    "YFinanceLoader",
    "validate_price_df",
    "get_eu_screener",
    "get_eu_smallcap_tickers",
    "get_eurostoxx50_tickers",
    "get_nasdaq100_screener",
    "get_nasdaq100_tickers",
    "get_screener_universe",
    "get_smallcap_tickers",
    "get_sp500_screener",
    "get_sp500_tickers",
    "get_us_smallcap_screener",
]

"""Shared cached resources for the Streamlit app.

Using @st.cache_resource means these objects are created once per server
process and reused across all sessions and page navigations.
"""

from __future__ import annotations

import streamlit as st

from config.settings import Settings
from data.cache.cache_manager import CacheManager
from data.data_service import DataService
from data.loaders.yfinance_loader import YFinanceLoader
from data.universes.eu_smallcap import get_eu_smallcap_tickers
from data.universes.eurostoxx50 import get_eurostoxx50_tickers
from data.universes.nasdaq100 import get_nasdaq100_tickers
from data.universes.screener import get_eu_screener, get_us_smallcap_screener
from data.universes.smallcap import get_smallcap_tickers
from data.universes.sp500 import get_sp500_tickers
from ranking.ranker import Ranker
from ranking.reversal_strategy import ReversalStrategy


@st.cache_resource
def get_settings() -> Settings:
    return Settings.default()


@st.cache_resource
def get_service() -> DataService:
    settings = get_settings()
    return DataService(YFinanceLoader(), CacheManager(settings.cache), settings)


@st.cache_resource
def get_ranker() -> Ranker:
    settings = get_settings()
    return Ranker.default(settings.ranking)


@st.cache_resource
def get_reversal_strategy() -> ReversalStrategy:
    settings = get_settings()
    return ReversalStrategy.default(settings.reversal)


@st.cache_resource
def get_universe(region: str = "US") -> list[str]:
    """Return ticker universe for the given region.

    Parameters
    ----------
    region:
        ``"US"``          — S&P 500 only
        ``"EU"``          — Euro Stoxx 50 + store europæiske caps
        ``"US+EU"``       — begge kombineret (duplikater fjernet)
        ``"NASDAQ-100"``  — NASDAQ-100 (Wikipedia, med fallback)
        ``"Small Cap"``   — ca. 250 likvide US small/mid-cap tickers
        ``"EU Small Cap"``— ca. 200 likvide EU small/mid-cap tickers
        ``"All US"``      — S&P 500 + NASDAQ-100 + Small Cap kombineret
        ``"All EU"``      — Euro Stoxx 50 + EU Small Cap kombineret
        ``"All"``         — alle ovenstaaende kombineret
        ``"Screener US"``  — live US small/mid caps via Yahoo screener
        ``"Screener EU"``  — live EU aktier via Yahoo screener
    """

    def _dedup(tickers: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for t in tickers:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out

    if region == "EU":
        return get_eurostoxx50_tickers()
    if region == "US+EU":
        return _dedup(get_sp500_tickers() + get_eurostoxx50_tickers())
    if region == "NASDAQ-100":
        return get_nasdaq100_tickers()
    if region == "Small Cap":
        return get_smallcap_tickers()
    if region == "EU Small Cap":
        return get_eu_smallcap_tickers()
    if region == "All US":
        return _dedup(
            get_sp500_tickers() + get_nasdaq100_tickers() + get_smallcap_tickers()
        )
    if region == "All EU":
        return _dedup(get_eurostoxx50_tickers() + get_eu_smallcap_tickers())
    if region == "All":
        return _dedup(
            get_sp500_tickers()
            + get_nasdaq100_tickers()
            + get_smallcap_tickers()
            + get_eurostoxx50_tickers()
            + get_eu_smallcap_tickers()
        )
    if region == "Screener US":
        return get_us_smallcap_screener(max_results=400)
    if region == "Screener EU":
        return get_eu_screener(max_results=300)
    return get_sp500_tickers()


@st.cache_resource(show_spinner="Henter firmanavne…")
def get_company_names(tickers: tuple[str, ...]) -> dict[str, str]:
    """Fetch longName for each ticker via yfinance. Cached for the server lifetime."""
    import yfinance as yf

    names: dict[str, str] = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            names[ticker] = info.get("longName") or info.get("shortName") or ticker
        except Exception:
            names[ticker] = ticker
    return names

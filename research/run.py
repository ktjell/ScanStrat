"""research/run.py

Hovedscript: vælg strategier, benchmark og periode — kør backtest.

    uv run python research/run.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

# Tilføj projektrod og research/ til sys.path
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "research"))

from engine.data import (
    get_eu_smallcap_tickers,
    get_eurostoxx50_tickers,
    get_nasdaq100_tickers,
    get_smallcap_tickers,
    get_sp500_tickers,
)
from runner import BacktestRunner
from strategies.daily_breakout import build as build_daily_breakout
from strategies.golden_cross import build as build_golden_cross
from strategies.high52w import build as build_high52w
from strategies.low_volatility import build as build_low_vol
from strategies.ml_ranker import build as build_ml_ranker
from strategies.momentum import build as build_momentum
from strategies.reversal import build as build_reversal
from strategies.weekly_momentum import build as build_weekly_momentum

# ------------------------------------------------------------------
# Konfiguration — tilpas her
# ------------------------------------------------------------------
# Univers-valg:
#   "US"          — S&P 500 (ca. 500 aktier)
#   "EU"          — Euro Stoxx 50 + store EU caps
#   "US+EU"       — S&P 500 + Euro Stoxx kombineret
#   "NASDAQ-100"  — NASDAQ top-100
#   "Small Cap"   — ca. 250 likvide US small/mid caps
#   "EU Small Cap"— ca. 200 likvide EU small/mid caps
#   "All US"      — S&P 500 + NASDAQ-100 + US Small Cap
#   "All EU"      — Euro Stoxx 50 + EU Small Cap
#   "All"         — alle ovenstaaende kombineret
UNIVERSE: str = "US+EU"  # bruges som standard-univers for strategier uden eget univers

BENCHMARK: str = "SPY"  # iShares MSCI World ETF (USD)
# alternativt: "IWDA.AS" (EUR, Euronext) | "SPY" | "QQQ"
TOP_N: int = 10
START: date = date(2020, 1, 1)
END: date = date.today()  # Henter alt tilgængeligt data — inkl. 2025-2026
COMMISSION: str = "saxo_classic"  # "saxo_classic" | "saxo_platinum" | "zero"

# Strategier der skal backtestes — tilfoej/fjern linjer her
# Tip: Tilfoej et 6. element (liste af tickers) for at give en strategi sit eget univers:
#   ("ML Ranker (US)", strat, rebalance_days, top_n, portfolio_usd, us_tickers)
_us = None  # udfyldes nedenfor efter _build_universe er defineret
_useu = None


def _get_strategies():
    name_us, strat_us, rd, tn = build_ml_ranker()
    name_eu, strat_eu, *_ = build_ml_ranker()
    us = _build_universe("US")
    eu = _build_universe("All EU")
    portfolio_usd = 72_500.0
    return [
        # Sammenlign begge universer side om side:
        (f"{name_us} (US)", strat_us, rd, tn, portfolio_usd, us),
        (f"{name_eu} (All EU)", strat_eu, rd, tn, portfolio_usd, eu),
        # Andre strategier (kommenter ud/ind efter behov):
        # build_momentum(),
        # build_weekly_momentum(),
        # build_golden_cross(),
        # build_reversal(),
        # build_low_vol(),
        # build_high52w(),
        # build_daily_breakout(),
        # build_ml_ranker(smart_rebalance=True),
    ]


# ------------------------------------------------------------------
def _build_universe(region: str) -> list[str]:
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
    return get_sp500_tickers()


if __name__ == "__main__":
    runner = BacktestRunner(
        top_n=TOP_N,
        commission=COMMISSION,
    )
    runner.run(
        strategies=_get_strategies(),
        universe=_build_universe(UNIVERSE),
        benchmark=BENCHMARK,
        start=START,
        end=END,
    )

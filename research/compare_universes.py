"""Sammenlign US vs US+EU univers head-to-head i ét plot.

uv run python research/compare_universes.py
"""

from __future__ import annotations
import sys
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "research"))

from engine.data import get_sp500_tickers, get_eurostoxx50_tickers
from runner import BacktestRunner
from strategies.ml_ranker import build as build_ml_ranker

START = date(2020, 1, 1)
END = date.today()
COMMISSION = "saxo_classic"

us = list(dict.fromkeys(get_sp500_tickers()))
eu = get_eurostoxx50_tickers()
us_eu = list(dict.fromkeys(us + eu))

# Omdøb strategierne så de kan skelnes i tabellen og plottet
name_us, strat_us, rd_us, tn_us = build_ml_ranker()
name_eu, strat_eu, rd_eu, tn_eu = build_ml_ranker()
name_us = "ML Ranker (US)"
name_eu = "ML Ranker (US+EU)"

runner = BacktestRunner(top_n=10, default_rebalance_days=5, commission=COMMISSION)

# Begge strategier i ét kald — eget univers som 6. element i tuple
runner.run(
    strategies=[
        (name_us, strat_us, rd_us, tn_us, runner._portfolio_usd, us),
        (name_eu, strat_eu, rd_eu, tn_eu, runner._portfolio_usd, eu),
    ],
    universe=us,  # bruges kun til data-hentning for strategier uden eget univers
    benchmark="SPY",
    start=START,
    end=END,
)

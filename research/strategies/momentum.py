"""research/strategies/momentum.py

Momentum-strategi baseret på Ranker (12M momentum, volatilitetsvægtning).
Bruges via build() i research/run.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from engine.config import Settings
from engine.ranking import Ranker


REBALANCE_DAYS: int = 30  # momentum: månedlig rebalancering


def build(rebalance_days: int = REBALANCE_DAYS) -> tuple[str, Ranker, int]:
    """Returnerer (navn, strategi-instans, rebalance_dage) klar til BacktestRunner."""
    settings = Settings.default()
    return "Momentum", Ranker.default(settings.ranking), rebalance_days

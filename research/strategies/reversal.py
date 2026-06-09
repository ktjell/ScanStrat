"""research/strategies/reversal.py

Contrarian/mean-reversion strategi: finder oversold aktier med
stigende volumen tæt på 52-ugers lav.
Bruges via build() i research/run.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from engine.config import ReversalSettings
from engine.ranking import ReversalStrategy

# ------------------------------------------------------------------
# Parametre — tilpas her
# ------------------------------------------------------------------
RSI_THRESHOLD: float = 35.0  # maks RSI (oversold grænse)
VOLUME_RATIO: float = 1.2  # min. volumen ift. 20d gennemsnit
LOOKBACK: int = 20  # dage til at måle negativt afkast
MAX_DIST_52W_LOW: float = 0.25  # maks afstand fra 52-ugers lav


REBALANCE_DAYS: int = 14  # reversal: kortere hold, hurtigere mean-reversion


def build(rebalance_days: int = REBALANCE_DAYS) -> tuple[str, ReversalStrategy, int]:
    """Returnerer (navn, strategi-instans, rebalance_dage) klar til BacktestRunner."""
    settings = ReversalSettings(
        rsi_threshold=RSI_THRESHOLD,
        volume_ratio_threshold=VOLUME_RATIO,
        return_lookback_days=LOOKBACK,
        max_dist_52w_low=MAX_DIST_52W_LOW,
    )
    return "Reversal", ReversalStrategy(settings), rebalance_days

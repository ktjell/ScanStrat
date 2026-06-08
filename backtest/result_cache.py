"""backtest/result_cache.py

Cache for backtest-resultater.

Noegle = SHA-256 af (strategi-navn, univers, start, end, top_n,
rebalance_days, commission).  Samme parametre => cache hit.

Data gemmes i data/backtest_cache/<key>/
  metrics.json        — dict med CAGR, Sharpe osv.
  equity_curve.parquet
  holdings.parquet
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import date
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).parent.parent / "research" / "data" / "backtest_cache"


# ---------------------------------------------------------------------------
# Noegle
# ---------------------------------------------------------------------------


def make_key(
    strategy_name: str,
    universe: list[str],
    start: date,
    end: date,
    top_n: int,
    rebalance_days: int,
    commission: str,
) -> str:
    """Returner en 16-karakter hex-noegle der unikt identificerer koeringen."""
    parts = "|".join(
        [
            strategy_name,
            ",".join(sorted(universe)),
            str(start),
            str(end),
            str(top_n),
            str(rebalance_days),
            commission,
        ]
    )
    return hashlib.sha256(parts.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Load / save
# ---------------------------------------------------------------------------


def load(key: str) -> "BacktestResult | None":  # noqa: F821
    from backtest.engine import BacktestResult

    path = _CACHE_DIR / key
    if not path.exists():
        return None
    try:
        metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
        equity_df = pd.read_parquet(path / "equity_curve.parquet")
        equity_curve = equity_df["value"]
        equity_curve.index.name = "date"
        holdings = pd.read_parquet(path / "holdings.parquet")
        logger.info("Backtest cache hit: %s", key)
        return BacktestResult(
            equity_curve=equity_curve, holdings=holdings, metrics=metrics
        )
    except Exception as exc:
        logger.warning("Kunne ikke laese backtest cache %s: %s", key, exc)
        return None


def save(key: str, result: "BacktestResult") -> None:  # noqa: F821
    path = _CACHE_DIR / key
    path.mkdir(parents=True, exist_ok=True)
    try:
        (path / "metrics.json").write_text(
            json.dumps(result.metrics, indent=2), encoding="utf-8"
        )
        pd.DataFrame({"value": result.equity_curve}).to_parquet(
            path / "equity_curve.parquet"
        )
        # holdings.tickers er en list — parquet/pyarrow haandterer det
        result.holdings.to_parquet(path / "holdings.parquet")
        logger.info("Backtest resultat gemt: %s", key)
    except Exception as exc:
        logger.warning("Kunne ikke gemme backtest resultat %s: %s", key, exc)

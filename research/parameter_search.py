"""research/run_parameter_search.py

Grid search over ReversalStrategy parametre.
Tilpas konstanterne herunder og kør:
    uv run python research/run_parameter_search.py
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import itertools
from datetime import date, timedelta

import pandas as pd

from engine.backtest import BacktestEngine, CommissionSchedule
from engine.config import ReversalSettings, Settings
from engine.data import CacheManager, DataService, YFinanceLoader, get_sp500_tickers
from engine.ranking import ReversalStrategy

# ------------------------------------------------------------------
# Parametre — tilpas her
# ------------------------------------------------------------------
START: date = date(2020, 1, 1)
END: date = date(2024, 12, 31)
TOP_N: int = 10
REBALANCE_FREQ: int = 30  # antal dage mellem rebalanceringer
SORT_METRIC: str = "cagr"  # "cagr" | "sharpe" | "max_drawdown" | "total_return"
TOP_RESULTS: int = 10

RSI_THRESHOLDS = [25, 30, 35, 40]
VOLUME_RATIOS = [1.2, 1.5, 2.0]
LOOKBACKS = [10, 20, 30]


def run(
    start: date = date(2020, 1, 1),
    end: date = date(2024, 12, 31),
    top_n: int = 10,
    rebalance_freq: int = 30,
    sort_metric: str = "cagr",
    top_results: int = 10,
    portfolio_size_usd: float = 72_500.0,
) -> None:
    settings = Settings.default()
    service = DataService(YFinanceLoader(), CacheManager(settings.cache), settings)

    universe = get_sp500_tickers()
    fetch_start = start - timedelta(days=365)

    print(
        f"Grid search: {len(RSI_THRESHOLDS)} RSI × {len(VOLUME_RATIOS)} vol × {len(LOOKBACKS)} lookback "
        f"= {len(RSI_THRESHOLDS) * len(VOLUME_RATIOS) * len(LOOKBACKS)} kombinationer"
    )
    print("Henter data…")
    data = service.get_batch(universe, fetch_start, end)

    freq_str = f"{rebalance_freq}D"
    commission = CommissionSchedule.saxo_classic()
    rows: list[dict] = []

    combos = list(itertools.product(RSI_THRESHOLDS, VOLUME_RATIOS, LOOKBACKS))
    for i, (rsi, vol, lookback) in enumerate(combos, 1):
        rev_settings = ReversalSettings(
            rsi_threshold=float(rsi),
            volume_ratio_threshold=vol,
            return_lookback_days=lookback,
            max_dist_52w_low=0.25,
        )
        strategy = ReversalStrategy(rev_settings)
        engine = BacktestEngine(
            ranker=strategy,  # type: ignore[arg-type]
            top_n=top_n,
            rebalance_freq=freq_str,
            commission_schedule=commission,
            portfolio_size_usd=portfolio_size_usd,
        )
        try:
            result = engine.run(data, start=start, end=end)
            m = result.metrics
            rows.append(
                {
                    "rsi_threshold": rsi,
                    "volume_ratio": vol,
                    "lookback_days": lookback,
                    "n_periods": len(result.holdings),
                    **m,
                }
            )
            if i % 5 == 0 or i == len(combos):
                print(f"  {i}/{len(combos)} kombinationer kørt…")
        except ValueError:
            rows.append(
                {
                    "rsi_threshold": rsi,
                    "volume_ratio": vol,
                    "lookback_days": lookback,
                    "n_periods": 0,
                    "cagr": float("nan"),
                    "sharpe": float("nan"),
                    "max_drawdown": float("nan"),
                    "total_return": float("nan"),
                    "win_rate": float("nan"),
                }
            )

    df = pd.DataFrame(rows)
    df = df.sort_values(sort_metric, ascending=False).head(top_results)

    print()
    print(f"Top {top_results} kombinationer (sorteret efter {sort_metric}):")
    print("=" * 80)
    print(df.to_string(index=False, float_format="{:.3f}".format))
    print("=" * 80)


if __name__ == "__main__":
    run(
        start=START,
        end=END,
        top_n=TOP_N,
        rebalance_freq=REBALANCE_FREQ,
        sort_metric=SORT_METRIC,
        top_results=TOP_RESULTS,
    )

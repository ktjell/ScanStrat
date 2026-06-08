"""engine.backtest — backtesting infrastructure."""

from backtest.commission import CommissionSchedule, ExchangeRate
from backtest.engine import BacktestEngine, BacktestResult
from backtest.metrics import compute_metrics, cagr, sharpe, max_drawdown, win_rate

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "CommissionSchedule",
    "ExchangeRate",
    "compute_metrics",
    "cagr",
    "sharpe",
    "max_drawdown",
    "win_rate",
]

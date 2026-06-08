from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from backtest.commission import CommissionSchedule
from backtest.metrics import compute_metrics
from ranking.ranker import Ranker

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Output from a single backtest run."""

    equity_curve: pd.Series
    """Daily portfolio value normalised to 1.0 at start."""

    holdings: pd.DataFrame
    """One row per rebalancing period: date, tickers held, equal-weight return."""

    metrics: dict[str, float]
    """Summary metrics: cagr, sharpe, max_drawdown, total_return."""

    def __str__(self) -> str:  # pragma: no cover
        m = self.metrics
        return (
            f"CAGR:          {m['cagr']:.1%}\n"
            f"Sharpe:        {m['sharpe']:.2f}\n"
            f"Max Drawdown:  {m['max_drawdown']:.1%}\n"
            f"Total Return:  {m['total_return']:.1%}"
        )


class BacktestEngine:
    """
    Walk-forward backtester.

    Strategy
    --------
    At each rebalancing date, rank all tickers *as_of* that date, buy the
    top-N equally weighted, hold until the next rebalancing date, then repeat.
    Returns are based on the actual close prices in *data*.

    Parameters
    ----------
    ranker:
        A configured Ranker instance.
    top_n:
        Number of top-ranked tickers to hold in each period.
    rebalance_freq:
        Pandas offset alias for rebalancing frequency (e.g. ``"ME"`` for
        month-end, ``"QE"`` for quarter-end).
    """

    def __init__(
        self,
        ranker: Ranker,
        top_n: int = 10,
        rebalance_freq: str = "ME",
        commission_schedule: CommissionSchedule | None = None,
        portfolio_size_usd: float = 100_000.0,
    ) -> None:
        if top_n < 1:
            raise ValueError("top_n must be >= 1")
        if portfolio_size_usd <= 0:
            raise ValueError("portfolio_size_usd must be > 0")
        self._ranker = ranker
        self._top_n = top_n
        self._rebalance_freq = rebalance_freq
        self._commission_schedule = commission_schedule or CommissionSchedule.zero()
        self._portfolio_size_usd = portfolio_size_usd

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        data: dict[str, pd.DataFrame],
        start: date,
        end: date,
    ) -> BacktestResult:
        """Run the backtest and return a BacktestResult.

        Parameters
        ----------
        data:
            OHLCV DataFrames keyed by ticker.  Each DataFrame must cover the
            full [start, end] range (fetched upfront by the caller).
        start:
            First rebalancing date (inclusive).
        end:
            Last date for which returns are measured.
        """
        rebalance_dates = self._rebalance_dates(start, end)
        if len(rebalance_dates) < 2:
            raise ValueError(
                "Date range too short — need at least two rebalancing dates."
            )

        holdings_rows: list[dict] = []
        portfolio_value = 1.0
        equity_points: list[tuple[date, float]] = [(start, portfolio_value)]
        prev_tickers: list[str] = []  # bruges af stateful strategier

        for i, rebal_date in enumerate(rebalance_dates[:-1]):
            next_date = rebalance_dates[i + 1]

            # Stateful strategi (fx GoldenCross): send nuværende beholdning ind
            if hasattr(self._ranker, "rebalance"):
                tickers = self._ranker.rebalance(prev_tickers, data, as_of=rebal_date)
                if not tickers:
                    logger.warning(
                        "No tickers from rebalance() on %s — skipping period.",
                        rebal_date,
                    )
                    equity_points.append((next_date, portfolio_value))
                    continue
            else:
                # Standard strategi: rank og tag top-N
                ranked = self._ranker.rank(data, as_of=rebal_date)
                if ranked.empty:
                    logger.warning(
                        "No tickers passed filters on %s — skipping period.", rebal_date
                    )
                    equity_points.append((next_date, portfolio_value))
                    continue
                tickers = ranked["ticker"].head(self._top_n).tolist()

            prev_tickers = tickers
            period_return = self._period_return(data, tickers, rebal_date, next_date)

            portfolio_value *= 1.0 + period_return
            equity_points.append((next_date, portfolio_value))

            holdings_rows.append(
                {
                    "date": rebal_date,
                    "tickers": tickers,
                    "n_held": len(tickers),
                    "period_return": period_return,
                }
            )

            logger.debug(
                "%s → %s | tickers=%s | period_return=%.2f%%",
                rebal_date,
                next_date,
                tickers,
                period_return * 100,
            )

        equity_curve = pd.Series(
            {pd.Timestamp(d): v for d, v in equity_points},
            name="portfolio_value",
        )
        equity_curve.index.name = "date"

        holdings = pd.DataFrame(holdings_rows)
        metrics = compute_metrics(equity_curve)

        logger.info(
            "Backtest complete: %d periods | CAGR=%.1f%% | Sharpe=%.2f | MaxDD=%.1f%%",
            len(holdings_rows),
            metrics["cagr"] * 100,
            metrics["sharpe"],
            metrics["max_drawdown"] * 100,
        )
        return BacktestResult(
            equity_curve=equity_curve, holdings=holdings, metrics=metrics
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rebalance_dates(self, start: date, end: date) -> list[date]:
        """Generate rebalancing dates between start and end (inclusive)."""
        idx = pd.date_range(start=start, end=end, freq=self._rebalance_freq)
        # Prepend start if not already there
        dates = [pd.Timestamp(start)] + list(idx)
        # Deduplicate and sort
        dates = sorted(set(dates))
        # Filter to [start, end]
        ts_start = pd.Timestamp(start)
        ts_end = pd.Timestamp(end)
        return [d.date() for d in dates if ts_start <= d <= ts_end]

    def _period_return(
        self,
        data: dict[str, pd.DataFrame],
        tickers: list[str],
        entry_date: date,
        exit_date: date,
    ) -> float:
        """Equal-weight return for *tickers* from entry_date close to exit_date close.

        Commission is deducted as a round-trip cost (buy + sell) per position.
        """
        returns: list[float] = []
        for ticker in tickers:
            df = data.get(ticker)
            if df is None or df.empty:
                continue
            close = df["close"]
            entry_close = self._nearest_close(close, entry_date)
            exit_close = self._nearest_close(close, exit_date)
            if entry_close is None or exit_close is None or entry_close == 0:
                continue
            gross = exit_close / entry_close - 1.0
            # Round-trip commission: pay once to buy, once to sell
            position_value = self._portfolio_size_usd / len(tickers)
            commission = self._commission_schedule.effective_pct(ticker, position_value)
            net = gross - 2 * commission
            returns.append(net)

        if not returns:
            return 0.0
        return float(sum(returns) / len(returns))

    @staticmethod
    def _nearest_close(close: pd.Series, target: date) -> float | None:
        """Return the close price on *target* or the next available trading day."""
        ts = pd.Timestamp(target)
        future = close.loc[close.index >= ts].dropna()
        if future.empty:
            return None
        return float(future.iloc[0])

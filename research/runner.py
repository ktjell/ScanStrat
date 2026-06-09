"""research/runner.py

BacktestRunner: orchestrerer backtest for N strategier + benchmark,
printer en metrics-tabel og gemmer et equity-curve plot som PNG.
"""

from __future__ import annotations

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from datetime import date, timedelta
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from engine.backtest import BacktestEngine, CommissionSchedule
from engine.config import Settings
from engine.data import CacheManager, DataService, YFinanceLoader
from backtest.result_cache import (
    load as _cache_load,
    make_key as _cache_key,
    save as _cache_save,
)
from backtest.regime import classify as _classify_regime, regime_periods, regime_metrics


# ------------------------------------------------------------------
# Hjælpefunktioner
# ------------------------------------------------------------------


def _build_service() -> DataService:
    settings = Settings.default()
    return DataService(YFinanceLoader(), CacheManager(settings.cache), settings)


def _bm_metrics(close: pd.Series, start: date, end: date) -> dict:
    """Buy-and-hold metrics for en kurs-serie."""
    s = close.loc[str(start) : str(end)].dropna()
    if len(s) < 2:
        nan = float("nan")
        return {"cagr": nan, "sharpe": nan, "max_drawdown": nan, "total_return": nan}
    curve = s / s.iloc[0]
    years = (pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25
    daily = s.pct_change().dropna()
    return {
        "cagr": float(curve.iloc[-1] ** (1 / years) - 1) if years > 0 else float("nan"),
        "sharpe": float(daily.mean() / daily.std() * np.sqrt(252))
        if daily.std() > 0
        else float("nan"),
        "max_drawdown": float((curve / curve.cummax() - 1).min()),
        "total_return": float(curve.iloc[-1] - 1),
    }


# ------------------------------------------------------------------
# BacktestRunner
# ------------------------------------------------------------------


class BacktestRunner:
    """
    Kører backtest for en liste af strategier og sammenligner med benchmark.

    Eksempel::

        runner = BacktestRunner(top_n=10, rebalance_days=30)
        runner.run(
            strategies=[("Momentum", ranker), ("Reversal", reversal)],
            universe=get_sp500_tickers(),
            benchmark="URTH",
            start=date(2020, 1, 1),
            end=date(2024, 12, 31),
        )
    """

    def __init__(
        self,
        top_n: int = 10,
        default_rebalance_days: int = 30,
        portfolio_usd: float = 72_500.0,
        commission: str = "saxo_classic",
    ) -> None:
        _map = {
            "saxo_classic": CommissionSchedule.saxo_classic(),
            "saxo_platinum": CommissionSchedule.saxo_platinum(),
            "zero": CommissionSchedule.zero(),
        }
        self._commission = _map.get(commission, CommissionSchedule.saxo_classic())
        self._commission_name = commission if commission in _map else "saxo_classic"
        self._top_n = top_n
        self._default_rebalance_days = default_rebalance_days
        self._portfolio_usd = portfolio_usd

    def run(
        self,
        strategies: list[tuple[str, Any] | tuple[str, Any, int]],
        universe: list[str],
        benchmark: str,
        start: date,
        end: date,
        use_cache: bool = True,
    ) -> None:
        service = _build_service()
        fetch_start = start - timedelta(days=365)

        print(
            f"\nUnivers: {len(universe)} aktier  |  {start} - {end}"
            f"  |  Top-{self._top_n}  |  kurtage: {self._commission_name}"
        )
        n_cached = service.count_cached(universe, fetch_start, end)
        n_fresh = len(universe) - n_cached
        print(
            f"Indlæser kursdata  ({n_cached} fra disk-cache"
            + (f", {n_fresh} hentes fra Yahoo" if n_fresh else "")
            + ")..."
        )
        data = service.get_batch(universe, fetch_start, end)
        print(f"  -> {len(data)} tickers med data")

        # --- kør hver strategi ---
        results: list[tuple[str, dict, pd.Series | None]] = []
        for entry in strategies:
            name, strategy = entry[0], entry[1]
            rebalance_days = (
                entry[2] if len(entry) >= 3 else self._default_rebalance_days
            )  # type: ignore[misc]
            top_n = entry[3] if len(entry) >= 4 else self._top_n  # type: ignore[misc]
            portfolio_usd = entry[4] if len(entry) >= 5 else self._portfolio_usd  # type: ignore[misc]
            # Valgfrit 6. element: eget univers for denne strategi
            strat_universe: list[str] | None = entry[5] if len(entry) >= 6 else None  # type: ignore[misc]
            freq = f"{rebalance_days}D"

            # Brug strategi-specifikt univers hvis angivet, ellers det globale
            if strat_universe is not None and strat_universe is not universe:
                strat_data = service.get_batch(strat_universe, fetch_start, end)
                effective_universe = strat_universe
            else:
                strat_data = data
                effective_universe = universe

            cache_key = (
                _cache_key(
                    name,
                    effective_universe,
                    start,
                    end,
                    top_n,
                    rebalance_days,
                    self._commission_name,
                )
                if use_cache
                else None
            )

            cached = _cache_load(cache_key) if cache_key else None
            if cached is not None:
                print(f"  {name}: hentet fra cache  ({rebalance_days}d, top-{top_n})")
                results.append(
                    (name, cached.metrics, cached.equity_curve, cached.holdings)
                )
                continue

            print(f"Korer {name}  ({rebalance_days}d rebalancering, top-{top_n})...")
            engine = BacktestEngine(
                ranker=strategy,  # type: ignore[arg-type]
                top_n=top_n,
                rebalance_freq=freq,
                commission_schedule=self._commission,
                portfolio_size_usd=portfolio_usd,
            )
            try:
                r = engine.run(strat_data, start=start, end=end)
                if cache_key:
                    _cache_save(cache_key, r)
                results.append((name, r.metrics, r.equity_curve, r.holdings))
            except ValueError as e:
                print(f"  FEJL {name}: {e}")
                results.append((name, {}, None, None))

        # --- benchmark ---
        print(f"Henter benchmark ({benchmark})...")
        bm_metrics: dict = {}
        bm_norm: pd.Series | None = None
        try:
            bm_df = service.get(benchmark, fetch_start, end)
            bm_close = bm_df["close"]
            bm_metrics = _bm_metrics(bm_close, start, end)
            sl = bm_close.loc[str(start) : str(end)].dropna()
            bm_norm = sl / sl.iloc[0]
        except Exception as e:
            print(f"  FEJL Benchmark ({benchmark}): {e}")

        # --- output ---
        regime: pd.Series | None = None
        if bm_norm is not None:
            try:
                bm_full = bm_df["close"].loc[str(start) : str(end)].dropna()
                regime = _classify_regime(bm_full)
            except Exception:
                pass
        self._print_table(results, benchmark, bm_metrics)
        self._print_regime_table(results, regime)
        self._plot(results, benchmark, bm_norm, regime, start, end)

        # --- Daily Breakout: udvidet statistik og eget plot ---
        for name, metrics, eq, holdings in results:
            if name == "Daily Breakout" and holdings is not None and not holdings.empty:
                self._print_breakout_stats(holdings)
                self._plot_daily_breakout(eq, holdings, start, end)

    # ------------------------------------------------------------------
    # Privat: tabel
    # ------------------------------------------------------------------

    def _print_table(
        self,
        results: list[tuple[str, dict, Any, Any]],
        bm_name: str,
        bm_metrics: dict,
    ) -> None:
        keys = ["cagr", "sharpe", "max_drawdown", "total_return", "win_rate"]
        labels = {
            "cagr": "CAGR",
            "sharpe": "Sharpe",
            "max_drawdown": "Max Drawdown",
            "total_return": "Total Return",
            "win_rate": "Win Rate",
        }
        cols = [(n, m) for n, m, *_ in results] + [(bm_name, bm_metrics)]
        # Beregn gns. positioner per strategi fra holdings
        avg_pos: dict[str, str] = {}
        for name, _, __, holdings in results:
            if (
                holdings is not None
                and not holdings.empty
                and "n_held" in holdings.columns
            ):
                avg_pos[name] = f"{holdings['n_held'].mean():.1f}"
            else:
                avg_pos[name] = "-"

        # Beregn handler/år (nye køb pr. år) fra holdings
        annual_trades: dict[str, str] = {}
        for name, _, __, holdings in results:
            if (
                holdings is not None
                and not holdings.empty
                and "tickers" in holdings.columns
                and len(holdings) >= 2
            ):
                tickers_list = holdings["tickers"].tolist()
                new_per_period = [
                    len(set(tickers_list[i]) - set(tickers_list[i - 1]))
                    for i in range(1, len(tickers_list))
                ]
                dates = pd.to_datetime(holdings["date"])
                avg_gap_days = (dates.iloc[-1] - dates.iloc[0]).days / max(
                    len(dates) - 1, 1
                )
                periods_per_year = 365.25 / max(avg_gap_days, 1)
                annual_trades[name] = (
                    f"{np.mean(new_per_period) * periods_per_year:.0f}"
                )
            else:
                annual_trades[name] = "-"
        cw = max(14, max(len(n) for n, _ in cols) + 2)
        w = 22 + cw * len(cols)
        sep = "=" * w

        print(f"\n{sep}")
        print(f"{'Metric':<22}" + "".join(f"{n:>{cw}}" for n, _ in cols))
        print("-" * w)
        for key in keys:
            row = f"{labels[key]:<22}"
            for name, m in cols:
                val = m.get(key, float("nan"))
                if key == "sharpe":
                    row += f" {val:>{cw - 1}.2f}"
                elif key == "win_rate" and name == bm_name:
                    row += f" {'-':>{cw - 1}}"
                else:
                    row += f" {val:>{cw - 1}.2%}"
            print(row)
        # Gns. positioner
        row = f"{'Gns. positioner':<22}"
        for name, _ in cols:
            v = avg_pos.get(name, "-")
            row += f" {v:>{cw - 1}}"
        print(row)
        # Handler/år
        row = f"{'Handler/år':<22}"
        for name, _ in cols:
            v = annual_trades.get(name, "-")
            row += f" {v:>{cw - 1}}"
        print(row)
        print(sep)

    def _print_regime_table(
        self,
        results: list[tuple[str, dict, Any, Any]],
        regime: "pd.Series | None",
    ) -> None:
        if regime is None:
            return
        bull_pct = (regime == "bull").mean()
        bear_pct = 1 - bull_pct

        names = [name for name, *_ in results if _ and _[1] is not None]
        if not names:
            return

        cw = max(14, max(len(n) for n in names) + 2)
        w = 22 + cw * len(names)
        sep = "=" * w

        print(f"\n{sep}")
        print(
            f"  Regime-analyse  (Bull: {bull_pct:.0%} af perioden  |  Bear: {bear_pct:.0%})"
        )
        print(sep)
        print(f"{'':22}" + "".join(f"{n:>{cw}}" for n in names))
        print("-" * w)

        for label, icon in (("bull", "Bull (groent)"), ("bear", "Bear (roedt)")):
            for metric, mlabel in (("cagr", "  CAGR"), ("max_drawdown", "  Max DD")):
                row = f"{icon + ' ' + mlabel:<22}"
                for name, _, eq, __ in results:
                    if eq is None or len(eq) == 0:
                        row += f" {'N/A':>{cw - 1}}"
                        continue
                    m = regime_metrics(eq, regime)
                    val = m.get(label, {}).get(metric, float("nan"))
                    row += (
                        f" {val:>{cw - 1}.1%}"
                        if not (val != val)
                        else f" {'N/A':>{cw - 1}}"
                    )
                print(row)
            print("-" * w)
        print(sep)

    def _plot(
        self,
        results: list[tuple[str, dict, pd.Series | None]],
        bm_name: str,
        bm_norm: pd.Series | None,
        regime: "pd.Series | None",
        start: date,
        end: date,
    ) -> None:
        fig, ax = plt.subplots(figsize=(13, 6))
        colors = plt.cm.tab10.colors  # type: ignore[attr-defined]

        # Farvede baggrundszoner: groen = bull, roed = bear
        if regime is not None:
            for seg_start, seg_end, label in regime_periods(regime):
                ax.axvspan(
                    seg_start,
                    seg_end,
                    color="#e8f5e9" if label == "bull" else "#fde8e8",
                    alpha=0.5,
                    linewidth=0,
                )

        for i, (name, _, eq, *__) in enumerate(results):
            if eq is None or len(eq) == 0:
                continue
            norm = eq / eq.iloc[0]
            norm.index = pd.to_datetime(norm.index)
            ax.plot(
                norm.index, norm.values, label=name, color=colors[i % 10], linewidth=2
            )

        if bm_norm is not None:
            bm_norm.index = pd.to_datetime(bm_norm.index)
            ax.plot(
                bm_norm.index,
                bm_norm.values,
                label=bm_name,
                color="dimgrey",
                linewidth=1.5,
                linestyle="--",
            )

        ax.axhline(1.0, color="lightgrey", linestyle=":", linewidth=1)
        ax.set_title(f"Equity curves  {start} - {end}", fontsize=13)
        ax.set_ylabel("Normaliseret (start = 1)")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Privat: Daily Breakout udvidet output
    # ------------------------------------------------------------------

    def _print_breakout_stats(self, holdings: pd.DataFrame) -> None:
        r = holdings["period_return"].dropna()
        if r.empty:
            return
        wins = r[r > 0]
        losses = r[r <= 0]
        profit_factor = (
            wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float("inf")
        )
        sep = "-" * 46
        print(f"\n{sep}")
        print(f"  Daily Breakout: per-periode statistik ({len(r)} handler)")
        print(sep)
        print(f"  {'Win rate (perioder):':<26} {len(wins) / len(r):.1%}")
        print(
            f"  {'Gns. gevinst:':<26} {wins.mean():.2%}"
            if len(wins)
            else "  Gns. gevinst: N/A"
        )
        print(
            f"  {'Gns. tab:':<26} {losses.mean():.2%}"
            if len(losses)
            else "  Gns. tab: N/A"
        )
        print(f"  {'Stoerste gevinst:':<26} {r.max():.2%}")
        print(f"  {'Stoerste tab:':<26} {r.min():.2%}")
        print(f"  {'Profit factor:':<26} {profit_factor:.2f}")
        print(sep)

    def _plot_daily_breakout(
        self,
        equity: pd.Series | None,
        holdings: pd.DataFrame,
        start: date,
        end: date,
    ) -> None:
        r = holdings.set_index("date")["period_return"].dropna()
        r.index = pd.to_datetime(r.index)

        fig, (ax1, ax2, ax3) = plt.subplots(
            3,
            1,
            figsize=(14, 11),
            gridspec_kw={"height_ratios": [2, 1.5, 1]},
        )

        # --- Equity curve ---
        if equity is not None and len(equity):
            norm = equity / equity.iloc[0]
            norm.index = pd.to_datetime(norm.index)
            ax1.plot(norm.index, norm.values, color="steelblue", linewidth=1.5)
            ax1.axhline(1.0, color="lightgrey", linestyle=":", linewidth=1)
        ax1.set_title(f"Daily Breakout  {start} - {end}", fontsize=12)
        ax1.set_ylabel("Equity (start = 1)")
        ax1.grid(True, alpha=0.3)

        # --- Per-dag bar chart ---
        colors = ["#2ca02c" if v > 0 else "#d62728" for v in r.values]
        ax2.bar(r.index, r.values * 100, color=colors, width=0.6)
        ax2.axhline(0, color="black", linewidth=0.8)
        ax2.set_ylabel("Afkast per dag (%)")
        ax2.grid(True, alpha=0.3, axis="y")

        # --- Histogram ---
        ax3.hist(
            r.values * 100,
            bins=40,
            color="steelblue",
            edgecolor="white",
            linewidth=0.4,
        )
        ax3.axvline(0, color="black", linewidth=0.8)
        ax3.axvline(
            r.mean() * 100,
            color="orange",
            linewidth=1.5,
            linestyle="--",
            label=f"Gns. {r.mean():.2%}",
        )
        ax3.set_xlabel("Afkast (%)")
        ax3.set_ylabel("Antal dage")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3, axis="y")

        fig.tight_layout()
        plt.show()

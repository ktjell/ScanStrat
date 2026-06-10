"""research/run_pead_backtest.py

PEAD (Post-Earnings Announcement Drift) backtest.

Simulerer strategien på historiske earnings-data fra S&P 500.
Fordi yfinance earnings-historik kun rækker ~2 år tilbage, er backtesten
begrænset til den tilgængelige periode.

Kør med:
    uv run python research/run_pead_backtest.py
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "research"))

from engine.data import get_sp500_tickers
from research.strategies.pead import HOLD_DAYS, SURPRISE_THRESHOLD

# ------------------------------------------------------------------
# Konfiguration
# ------------------------------------------------------------------
START: date = date(2023, 1, 1)  # yfinance earnings-historik er begrænset
END: date = date.today()
HOLD_DAYS_BT: int = HOLD_DAYS  # brug samme som live-strategi
SURPRISE_THRESHOLD_BT: float = SURPRISE_THRESHOLD
MAX_POSITIONS: int = 10  # max samtidige positioner
COMMISSION_PCT: float = 0.001  # 0.1% per handel (Saxo approx)
PORTFOLIO_START: float = 50_000.0  # DKK startkapital

OUTPUT_DIR = Path(_ROOT / "research" / "data")
OUTPUT_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------------
# Data helpers
# ------------------------------------------------------------------


def _fetch_price_history(
    tickers: list[str], start: date, end: date
) -> dict[str, pd.Series]:
    """Hent daglige close-kurser for alle tickers."""
    print(f"Henter kursdata for {len(tickers)} tickers...")
    try:
        raw = yf.download(
            tickers,
            start=str(start - timedelta(days=30)),
            end=str(end + timedelta(days=5)),
            auto_adjust=True,
            progress=False,
        )
        if raw.empty:
            return {}
        close = raw["Close"] if "Close" in raw.columns else raw
        if isinstance(close, pd.Series):
            close = close.to_frame(name=tickers[0])
        return {str(t): close[t].dropna() for t in tickers if t in close.columns}
    except Exception as e:
        print(f"Fejl ved hentning af kursdata: {e}")
        return {}


def _fetch_all_earnings(tickers: list[str]) -> pd.DataFrame:
    """
    Hent historiske earnings-data for alle tickers.
    Returnerer DataFrame med: ticker, earnings_date, actual_eps, estimated_eps, surprise_pct
    """
    print(
        f"Henter earnings-data for {len(tickers)} tickers (dette tager et stykke tid)..."
    )
    rows = []
    for i, ticker in enumerate(tickers):
        if i % 50 == 0:
            print(f"  {i}/{len(tickers)}...")
        try:
            t = yf.Ticker(ticker)
            ed = t.earnings_dates
            if ed is None or ed.empty:
                continue

            ed = ed.copy()
            if ed.index.tz is not None:
                ed.index = ed.index.tz_localize(None)
            ed.index = pd.to_datetime(ed.index)

            for dt, row in ed.iterrows():
                actual = row.get("Reported EPS")
                estimated = row.get("EPS Estimate")
                if pd.isna(actual) or pd.isna(estimated) or estimated == 0:
                    continue
                surprise_pct = float((actual - estimated) / abs(estimated))
                rows.append(
                    {
                        "ticker": ticker,
                        "earnings_date": dt.date(),
                        "actual_eps": float(actual),
                        "estimated_eps": float(estimated),
                        "surprise_pct": surprise_pct,
                    }
                )
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = (
        df[(df["earnings_date"] >= START) & (df["earnings_date"] <= END)]
        .sort_values("earnings_date")
        .reset_index(drop=True)
    )
    return df


# ------------------------------------------------------------------
# Backtest engine
# ------------------------------------------------------------------


def run_backtest(
    earnings_df: pd.DataFrame,
    prices: dict[str, pd.Series],
    benchmark_prices: pd.Series,
) -> dict:
    """
    Event-drevet backtest af PEAD-strategien.

    For hvert earnings-event med positiv surprise:
      - Entry: næste handelsdag efter earnings
      - Exit: efter HOLD_DAYS_BT handelsdage
    """
    if earnings_df.empty:
        return {"error": "Ingen earnings-data"}

    # Filtrer: kun positive surprises over threshold
    signals = earnings_df[earnings_df["surprise_pct"] >= SURPRISE_THRESHOLD_BT].copy()
    signals = signals.sort_values("earnings_date").reset_index(drop=True)
    print(f"Antal signals: {len(signals)} (af {len(earnings_df)} earnings events)")

    # Byg trade-liste
    trades = []
    for _, row in signals.iterrows():
        ticker = row["ticker"]
        if ticker not in prices:
            continue

        price_series = prices[ticker]
        earning_dt = pd.Timestamp(row["earnings_date"])

        # Find næste handelsdag efter earnings
        future = price_series[price_series.index > earning_dt]
        if future.empty:
            continue
        entry_dt = future.index[0]
        entry_price = float(future.iloc[0])

        # Find exit: HOLD_DAYS_BT handelsdage frem
        future_from_entry = price_series[price_series.index > entry_dt]
        if len(future_from_entry) < HOLD_DAYS_BT:
            if future_from_entry.empty:
                continue
            exit_dt = future_from_entry.index[-1]
            exit_price = float(future_from_entry.iloc[-1])
        else:
            exit_dt = future_from_entry.index[HOLD_DAYS_BT - 1]
            exit_price = float(future_from_entry.iloc[HOLD_DAYS_BT - 1])

        gross_return = (exit_price / entry_price) - 1
        net_return = gross_return - 2 * COMMISSION_PCT  # køb + salg

        trades.append(
            {
                "ticker": ticker,
                "earnings_date": row["earnings_date"],
                "surprise_pct": round(row["surprise_pct"] * 100, 2),
                "entry_date": entry_dt.date(),
                "entry_price": round(entry_price, 4),
                "exit_date": exit_dt.date(),
                "exit_price": round(exit_price, 4),
                "gross_return_pct": round(gross_return * 100, 2),
                "net_return_pct": round(net_return * 100, 2),
            }
        )

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return {"error": "Ingen trades genereret"}

    # --- Equity curve ---
    # Simuler portfolio: del kapital ligeligt mellem op til MAX_POSITIONS aktive positioner
    all_dates = pd.date_range(start=str(START), end=str(END), freq="B")
    equity = pd.Series(PORTFOLIO_START, index=all_dates)

    # Enkel simulering: hvert trade bidrager med PORTFOLIO_START / MAX_POSITIONS
    position_size = PORTFOLIO_START / MAX_POSITIONS
    pnl_by_exit = {}
    for _, t in trades_df.iterrows():
        exit_key = pd.Timestamp(t["exit_date"])
        pnl = position_size * t["net_return_pct"] / 100
        pnl_by_exit[exit_key] = pnl_by_exit.get(exit_key, 0.0) + pnl

    for i in range(1, len(equity)):
        dt = equity.index[i]
        equity.iloc[i] = equity.iloc[i - 1] + pnl_by_exit.get(dt, 0.0)

    # Benchmark equity curve
    bm_period = benchmark_prices[str(START) : str(END)].dropna()
    if not bm_period.empty:
        bm_equity = bm_period / bm_period.iloc[0] * PORTFOLIO_START
    else:
        bm_equity = None

    # --- Metrics ---
    n_trades = len(trades_df)
    n_wins = (trades_df["net_return_pct"] > 0).sum()
    win_rate = n_wins / n_trades if n_trades > 0 else 0
    avg_return = trades_df["net_return_pct"].mean()
    median_return = trades_df["net_return_pct"].median()
    best = trades_df["net_return_pct"].max()
    worst = trades_df["net_return_pct"].min()

    years = (END - START).days / 365.25
    total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else 0

    daily_ret = equity.pct_change().dropna()
    sharpe = (
        (daily_ret.mean() / daily_ret.std() * np.sqrt(252))
        if daily_ret.std() > 0
        else 0
    )
    max_dd = ((equity / equity.cummax()) - 1).min()

    print(f"\n{'=' * 55}")
    print(f"  PEAD Backtest  |  {START} - {END}")
    print(f"{'=' * 55}")
    print(f"  Antal trades:     {n_trades}")
    print(f"  Win rate:         {win_rate:.1%}")
    print(f"  Snit afkast/trade:{avg_return:+.2f}%")
    print(f"  Median:           {median_return:+.2f}%")
    print(f"  Bedste trade:     {best:+.2f}%")
    print(f"  Værste trade:     {worst:+.2f}%")
    print(f"  Total afkast:     {total_return:+.1%}")
    print(f"  CAGR:             {cagr:+.1%}")
    print(f"  Sharpe:           {sharpe:.2f}")
    print(f"  Max drawdown:     {max_dd:.1%}")
    print(f"{'=' * 55}\n")

    # --- Plot ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(f"PEAD Backtest  ({START} → {END})", fontsize=14)

    ax1 = axes[0]
    ax1.plot(equity.index, equity.values, label="PEAD", color="steelblue", linewidth=2)
    if bm_equity is not None:
        ax1.plot(
            bm_equity.index,
            bm_equity.values,
            label="SPY (B&H)",
            color="gray",
            linewidth=1.5,
            linestyle="--",
            alpha=0.8,
        )
    ax1.set_ylabel("Portfolio (DKK)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2 = axes[1]
    colors = ["green" if r > 0 else "red" for r in trades_df["net_return_pct"]]
    ax2.bar(range(len(trades_df)), trades_df["net_return_pct"], color=colors, alpha=0.7)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("Trade #")
    ax2.set_ylabel("Afkast per trade (%)")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plot_path = OUTPUT_DIR / "pead_backtest.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot gemt: {plot_path}")

    # Gem trades til JSON
    trades_path = OUTPUT_DIR / "pead_backtest_trades.json"
    trades_df_serializable = trades_df.copy()
    for col in ["earnings_date", "entry_date", "exit_date"]:
        trades_df_serializable[col] = trades_df_serializable[col].astype(str)

    with open(trades_path, "w", encoding="utf-8") as f:
        json.dump(trades_df_serializable.to_dict(orient="records"), f, indent=2)
    print(f"Trades gemt: {trades_path}")

    return {
        "n_trades": n_trades,
        "win_rate": win_rate,
        "avg_return_pct": avg_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "trades": trades_df,
        "equity": equity,
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main() -> None:
    tickers = get_sp500_tickers()

    # Hent kursdata
    prices = _fetch_price_history(tickers, START, END)

    # Hent benchmark
    print("Henter SPY...")
    spy_raw = yf.download(
        "SPY", start=str(START), end=str(END), auto_adjust=True, progress=False
    )
    bm_prices = spy_raw["Close"] if "Close" in spy_raw.columns else spy_raw.iloc[:, 0]

    # Hent earnings-data
    earnings = _fetch_all_earnings(tickers)

    if earnings.empty:
        print(
            "FEJL: Ingen earnings-data hentet. yfinance har muligvis ikke historik for denne periode."
        )
        return

    print(
        f"\nFærdig — {len(earnings)} earnings events for {earnings['ticker'].nunique()} aktier"
    )
    run_backtest(earnings, prices, bm_prices)


if __name__ == "__main__":
    main()

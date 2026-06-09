"""research/run_portfolio.py

Morgenrapport: hold/saelg-signaler for din portefoelge.

Kores dagligt (eller naar som helst):
    uv run python research/run_portfolio.py
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from engine.data import CacheManager, DataService, YFinanceLoader
from engine.config import Settings
from portfolio.holdings import HOLDINGS
from portfolio.signals import compute_signals, Signal
from portfolio.charts import plot_holding
import matplotlib.pyplot as plt


_ACTION_ICON = {
    "SAELG": "!! SAELG    !!",
    "ADVARSEL": "?  ADVARSEL  ?",
    "BEHOLD": "   BEHOLD    ",
    "INGEN DATA": "   INGEN DATA",
}


def _fmt(s: Signal) -> None:
    icon = _ACTION_ICON.get(s.action, s.action)
    print(f"  [{icon}]  {s.ticker:<12}  {s.name}")
    for r in s.reasons:
        print(f"             - {r}")
    if not any(x in s.action for x in ("DATA",)):
        print(
            f"             Kurs: {s.close:.2f}  "
            f"SMA50: {s.sma50:.2f}  "
            f"SMA200: {s.sma200:.2f}  "
            f"RSI: {s.rsi:.1f}  "
            f"4u: {s.return_4w:.1%}"
        )


def main() -> None:
    settings = Settings.default()
    service = DataService(YFinanceLoader(), CacheManager(settings.cache), settings)

    tickers = [h.ticker for h in HOLDINGS]
    end = date.today()
    start = end - timedelta(days=400)

    print(f"\nHenter kursdata for {len(tickers)} aktier...")
    data = service.get_batch(tickers, start, end)
    print(f"Data hentet for {len(data)} aktier.\n")

    signals = compute_signals(data, HOLDINGS)

    sep = "=" * 70
    print(sep)
    print(f"  PORTFOLIO RAPPORT  —  {end}")
    print(sep)

    sell = [s for s in signals if s.action == "SAELG"]
    warn = [s for s in signals if s.action == "ADVARSEL"]
    hold = [s for s in signals if s.action == "BEHOLD"]
    no_data = [s for s in signals if s.action == "INGEN DATA"]

    if sell:
        print(f"\n  SAELG ({len(sell)} aktie(r)):")
        for s in sell:
            _fmt(s)

    if warn:
        print(f"\n  ADVARSEL ({len(warn)} aktie(r)):")
        for s in warn:
            _fmt(s)

    if hold:
        print(f"\n  BEHOLD ({len(hold)} aktie(r)):")
        for s in hold:
            _fmt(s)

    if no_data:
        print(f"\n  INGEN DATA ({len(no_data)} aktie(r)):")
        for s in no_data:
            print(f"    {s.ticker:<12}  {s.name}")

    print(f"\n{sep}\n")

    # --- plots ---
    print("Genererer charts...")
    for s in signals:
        if s.action == "INGEN DATA":
            continue
        df = data.get(s.ticker)
        if df is None or df.empty:
            continue
        fig = plot_holding(df, s)
        plt.show()


if __name__ == "__main__":
    main()

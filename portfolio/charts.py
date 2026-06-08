"""portfolio/charts.py

Genererer signal-charts for portefoelge-aktier.

Hvert chart viser:
  - Priskurve med SMA50 (blaa) og SMA200 (orange)
  - Baggrundsfarve: roed = death cross, groen = golden cross
  - RSI-subplot med overkobt (>75) og oversolgt (<30) zoner
  - Signal-annotation (SAELG / ADVARSEL / BEHOLD) oeverst
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from portfolio.signals import Signal


_ACTION_COLOR = {
    "SAELG": "#d62728",
    "ADVARSEL": "#ff7f0e",
    "BEHOLD": "#2ca02c",
    "INGEN DATA": "#aaaaaa",
}


def _rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    return 100 - 100 / (1 + rs)


def plot_holding(
    df: pd.DataFrame,
    signal: Signal,
    lookback_days: int = 252,
) -> plt.Figure:
    """Returner en matplotlib Figure med pris + RSI chart for én aktie."""
    df = df.sort_index().copy()
    df.index = pd.to_datetime(df.index)
    close = df["close"].dropna()

    if len(close) > lookback_days:
        close = close.iloc[-lookback_days:]
        df = df.iloc[-lookback_days:]

    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    rsi = _rsi_series(close)

    action_color = _ACTION_COLOR.get(signal.action, "#aaaaaa")

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(13, 7),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    # ------------------------------------------------------------------
    # Prispanel
    # ------------------------------------------------------------------

    # Baggrundsfarve: roed = death cross, groen = golden cross
    if len(sma50.dropna()) > 0 and len(sma200.dropna()) > 0:
        both = sma50.align(sma200, join="inner")[0].index
        death = sma50.reindex(both) < sma200.reindex(both)
        prev = None
        seg_start = None
        for ts, is_death in death.items():
            if prev is None:
                prev = is_death
                seg_start = ts
                continue
            if is_death != prev:
                color = "#ffe0e0" if prev else "#e0f0e0"
                ax1.axvspan(seg_start, ts, color=color, alpha=0.4, linewidth=0)
                seg_start = ts
                prev = is_death
        if seg_start is not None:
            color = "#ffe0e0" if prev else "#e0f0e0"
            ax1.axvspan(seg_start, close.index[-1], color=color, alpha=0.4, linewidth=0)

    ax1.plot(close.index, close.values, color="steelblue", linewidth=1.5, label="Kurs")
    ax1.plot(
        sma50.index,
        sma50.values,
        color="#1f77b4",
        linewidth=1.2,
        linestyle="--",
        label="SMA50",
    )
    ax1.plot(
        sma200.index,
        sma200.values,
        color="#ff7f0e",
        linewidth=1.2,
        linestyle="--",
        label="SMA200",
    )

    # Signal-annotation
    ax1.annotate(
        f"  {signal.action}",
        xy=(close.index[-1], close.iloc[-1]),
        xytext=(0.98, 0.96),
        textcoords="axes fraction",
        ha="right",
        va="top",
        fontsize=12,
        fontweight="bold",
        color=action_color,
    )

    ax1.set_title(f"{signal.ticker}  —  {signal.name}", fontsize=12)
    ax1.set_ylabel("Kurs")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # RSI-panel
    # ------------------------------------------------------------------
    ax2.plot(rsi.index, rsi.values, color="purple", linewidth=1.2)
    ax2.axhline(75, color="#d62728", linewidth=0.8, linestyle="--", label="Overkobt 75")
    ax2.axhline(
        30, color="#2ca02c", linewidth=0.8, linestyle="--", label="Oversolgt 30"
    )
    ax2.fill_between(
        rsi.index,
        75,
        rsi.values.clip(None, 100),
        where=(rsi > 75),
        color="#d62728",
        alpha=0.15,
    )
    ax2.fill_between(
        rsi.index,
        rsi.values.clip(0, None),
        30,
        where=(rsi < 30),
        color="#2ca02c",
        alpha=0.15,
    )
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("RSI")
    ax2.legend(loc="upper left", fontsize=7)
    ax2.grid(True, alpha=0.3)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    fig.autofmt_xdate(rotation=30, ha="right")

    # Signal-årsager som figur-undertitel
    subtitle = "  ·  ".join(signal.reasons[:3])
    fig.text(0.5, 0.01, subtitle, ha="center", fontsize=8, color=action_color)

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    return fig

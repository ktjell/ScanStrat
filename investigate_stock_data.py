from datetime import date
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from backtest.commission import CommissionSchedule
from backtest.engine import BacktestEngine
from config.settings import Settings
from data.cache.cache_manager import CacheManager
from data.loaders.yfinance_loader import YFinanceLoader
from data.data_service import DataService
from data.universes.sp500 import get_sp500_tickers
from features.feature_engine import FeatureEngine
from features.oscillators import RSI14
from features.trend import SMA50, SMA200
from ranking.ranker import Ranker


TICKER = "MSFT"
START = date(2023, 1, 1)
END = date.today()

settings = Settings.default()
service = DataService(YFinanceLoader(), CacheManager(settings.cache), settings)

ranker = Ranker.default(settings.ranking)
data = service.get_batch(["AAPL", "MSFT", "NVDA", "META"], START, END)
result = ranker.rank(data)
print(result[["rank", "ticker", "score"]].to_string(index=False))

df = service.get(TICKER, START, END)
engine = FeatureEngine.default()
row = engine.compute_row(TICKER, df)

# --- RSI rolling series (for plot) ---
close = df["close"]
delta = close.diff()
alpha = 1.0 / 14
gain = delta.clip(lower=0).ewm(alpha=alpha, adjust=False, min_periods=14).mean()
loss = (-delta).clip(lower=0).ewm(alpha=alpha, adjust=False, min_periods=14).mean()
rsi_series = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

# --- SMA rolling series ---
sma50_series = close.rolling(50).mean()
sma200_series = close.rolling(200).mean()

# ---- Backtest engine demo ----
BT_START = date(2020, 1, 1)
BT_END = date(2024, 12, 31)
TOP_N = 10
PORTFOLIO_SIZE_USD = 100_000.0

# S&P 500 univers (hentes fra Wikipedia, fallback til ~100 store caps)
universe = get_sp500_tickers()
print(f"Univers: {len(universe)} aktier")

# ---------------------------------------------------------------
# Plot 1: Backtest equity curve  +  SPY benchmark
# ---------------------------------------------------------------
print("Henter kursdata for universet (kan tage lidt tid første gang)...")
backtest_data = service.get_batch(universe, BT_START, END)

# SPY buy-and-hold benchmark
spy_df = service.get("SPY", BT_START, END)
spy_close = spy_df["close"].loc[str(BT_START) : str(BT_END)]
spy_curve = spy_close / spy_close.iloc[0]  # normaliseret til 1.0

bt_engine = BacktestEngine(
    ranker=Ranker.default(settings.ranking),
    top_n=TOP_N,
    rebalance_freq="ME",
    commission_schedule=CommissionSchedule.saxo_classic(),
    portfolio_size_usd=PORTFOLIO_SIZE_USD,
)
bt_result = bt_engine.run(backtest_data, start=BT_START, end=BT_END)

fig_bt, ax_bt = plt.subplots(figsize=(12, 5))
ax_bt.plot(
    bt_result.equity_curve.index,
    bt_result.equity_curve.values,
    color="steelblue",
    linewidth=1.5,
    label=f"Strategi (top-{TOP_N}, månedlig rebalancering)",
)
# SPY benchmark
ax_bt.plot(
    spy_curve.index,
    spy_curve.values,
    color="darkorange",
    linewidth=1.2,
    linestyle="--",
    label="SPY (buy & hold)",
)
ax_bt.axhline(
    1.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.6, label="Start (1.0)"
)
ax_bt.fill_between(
    bt_result.equity_curve.index,
    1.0,
    bt_result.equity_curve.values,
    where=(bt_result.equity_curve.values >= 1.0),
    alpha=0.1,
    color="green",
)
ax_bt.fill_between(
    bt_result.equity_curve.index,
    1.0,
    bt_result.equity_curve.values,
    where=(bt_result.equity_curve.values < 1.0),
    alpha=0.15,
    color="red",
)
m = bt_result.metrics
spy_cagr = (
    spy_curve.iloc[-1]
    ** (1 / ((pd.Timestamp(BT_END) - pd.Timestamp(BT_START)).days / 365.25))
    - 1
)

# --- Metrics tabel ---
from backtest.metrics import compute_metrics, win_rate as _win_rate

spy_metrics = compute_metrics(spy_curve)
metrics_table = pd.DataFrame(
    {
        "Metric": [
            "CAGR",
            "Sharpe",
            "Max Drawdown",
            "Total Return",
            "Win Rate",
            "Kurtage (r/t)",
        ],
        f"Strategi (top-{TOP_N})": [
            f"{m['cagr']:.1%}",
            f"{m['sharpe']:.2f}",
            f"{m['max_drawdown']:.1%}",
            f"{m['total_return']:.1%}",
            f"{m['win_rate']:.0%}",
            "Saxo Classic (0.08%, min 1 USD)",
        ],
        "SPY (buy & hold)": [
            f"{spy_metrics['cagr']:.1%}",
            f"{spy_metrics['sharpe']:.2f}",
            f"{spy_metrics['max_drawdown']:.1%}",
            f"{spy_metrics['total_return']:.1%}",
            f"{spy_metrics['win_rate']:.0%}",
            "—",
        ],
    }
).set_index("Metric")
print(f"\n{'─' * 45}")
print(f"  Backtest {BT_START.year}–{BT_END.year}  |  univers: {len(universe)} aktier")
print(f"{'─' * 45}")
print(metrics_table.to_string())
print(f"{'─' * 45}\n")

ax_bt.set_title(
    f"Backtest {BT_START.year}–{BT_END.year}  |  "
    f"CAGR {m['cagr']:.1%}  Sharpe {m['sharpe']:.2f}  Max DD {m['max_drawdown']:.1%}  Win Rate {m['win_rate']:.0%}"
    f"  |  SPY CAGR {spy_cagr:.1%}",
    fontsize=10,
)
ax_bt.set_ylabel("Porteføljeværdi (normaliseret)")

# Højre y-akse: afkast i procent
ax_bt_pct = ax_bt.twinx()
ax_bt_pct.set_ylim(
    (ax_bt.get_ylim()[0] - 1) * 100,
    (ax_bt.get_ylim()[1] - 1) * 100,
)
ax_bt_pct.set_ylabel("Afkast (%)")
ax_bt_pct.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:+.0f}%"))

ax_bt.legend(fontsize=9)
ax_bt.grid(axis="y", alpha=0.3)
fig_bt.tight_layout()

# ---------------------------------------------------------------
# Plot 2: 3 panels — Price+SMA | RSI | Feature summary
# ---------------------------------------------------------------
fig = plt.figure(figsize=(14, 10))
fig.suptitle(f"{TICKER}  |  {END}", fontsize=14, fontweight="bold")
gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1.5, 1.5], hspace=0.4)

# Panel 1 – Price + SMAs
ax1 = fig.add_subplot(gs[0])
ax1.plot(df.index, close, label="Close", color="steelblue", linewidth=1.2)
ax1.plot(
    df.index,
    sma50_series,
    label="SMA 50",
    color="orange",
    linewidth=1.0,
    linestyle="--",
)
ax1.plot(
    df.index, sma200_series, label="SMA 200", color="red", linewidth=1.0, linestyle="--"
)
ax1.set_ylabel("Price (USD)")
ax1.legend(fontsize=8)
ax1.set_title("Kurs")

# Panel 2 – RSI
ax2 = fig.add_subplot(gs[1], sharex=ax1)
ax2.plot(df.index, rsi_series, color="purple", linewidth=1.0)
ax2.axhline(70, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
ax2.axhline(30, color="green", linestyle="--", linewidth=0.8, alpha=0.7)
ax2.fill_between(
    df.index, rsi_series, 70, where=(rsi_series >= 70), alpha=0.15, color="red"
)
ax2.fill_between(
    df.index, rsi_series, 30, where=(rsi_series <= 30), alpha=0.15, color="green"
)
ax2.set_ylim(0, 100)
ax2.set_ylabel("RSI 14")
ax2.set_title("RSI")

# Panel 3 – Feature snapshot (text table)
ax3 = fig.add_subplot(gs[2])
ax3.axis("off")
labels = [k for k in row if k != "ticker"]
values = [
    f"{row[k]:.4f}" if not (isinstance(row[k], float) and np.isnan(row[k])) else "n/a"
    for k in labels
]
col_labels = ["Feature", "Værdi"]
table_data = list(zip(labels, values))
table = ax3.table(
    cellText=table_data,
    colLabels=col_labels,
    cellLoc="center",
    loc="center",
    bbox=[0, 0, 1, 1],
)
table.auto_set_font_size(False)
table.set_fontsize(8)
ax3.set_title("Feature snapshot (seneste dato)", pad=4)

plt.tight_layout()
plt.show()

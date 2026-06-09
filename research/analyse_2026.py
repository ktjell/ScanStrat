"""research/analyse_2026.py

Detaljeret 2026-analyse af ML Ranker:
  - Hvad er købt/solgt hvornår
  - Afkast per dag/uge/måned vs SPY
  - Vindere og tabere
  - Diverse kvalitetsmetrikker

    uv run python research/analyse_2026.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "research"))

import json
import numpy as np
import pandas as pd
import yfinance as yf

from backtest.result_cache import make_key as _cache_key
from engine.data import (
    get_eurostoxx50_tickers,
    get_sp500_tickers,
)

# -----------------------------------------------------------------------
# Find cache-nøgle — samme parametre som run.py
# -----------------------------------------------------------------------
UNIVERSE: str = "US"
START = date(2020, 1, 1)
END = date.today()
TOP_N = 15
REBALANCE_DAYS = 5
COMMISSION = "saxo_classic"
STRATEGY_NAME = "ML Ranker"
YEAR = 2026

_sp500 = get_sp500_tickers()
_universe = list(dict.fromkeys(_sp500))

cache_key = _cache_key(
    STRATEGY_NAME, _universe, START, END, TOP_N, REBALANCE_DAYS, COMMISSION
)
# Søg i både research/data/backtest_cache og rod/data/backtest_cache
_search_roots = [
    Path("data/backtest_cache"),  # relativ (research/)
    _ROOT / "data" / "backtest_cache",  # projektrod
]
cache_dir = None
for _root in _search_roots:
    candidate = _root / cache_key
    if candidate.exists():
        cache_dir = candidate
        break

if cache_dir is None:
    # Fallback: tag nyeste på tværs af begge steder
    all_candidates = []
    for _root in _search_roots:
        if _root.exists():
            all_candidates.extend(_root.iterdir())
    if not all_candidates:
        print("Ingen cache fundet — kør run.py først.")
        sys.exit(1)
    cache_dir = max(all_candidates, key=lambda p: p.stat().st_mtime)
    print(f"Bruger cache (fallback): {cache_dir.name}")
else:
    print(f"Bruger cache: {cache_dir.name}")

holdings = pd.read_parquet(cache_dir / "holdings.parquet")
equity_raw = pd.read_parquet(cache_dir / "equity_curve.parquet")
metrics_full = json.loads((cache_dir / "metrics.json").read_text())

# -----------------------------------------------------------------------
# Filtrér til YEAR+
# -----------------------------------------------------------------------
holdings["date"] = pd.to_datetime(holdings["date"])
holdings_2026 = holdings[holdings["date"].dt.year >= YEAR].copy()
holdings_2026 = holdings_2026.reset_index(drop=True)

equity = equity_raw["value"] if "value" in equity_raw.columns else equity_raw.iloc[:, 0]
equity.index = pd.to_datetime(equity.index)
equity_2026 = equity[equity.index.year >= YEAR]

# -----------------------------------------------------------------------
# Hent SPY
# -----------------------------------------------------------------------
spy_raw = yf.download(
    "SPY", start=f"{YEAR}-01-01", end=str(END), auto_adjust=True, progress=False
)
spy_close = spy_raw["Close"].squeeze()
spy_ret = spy_close.pct_change().dropna()


# -----------------------------------------------------------------------
# AFKAST: daglig / ugentlig / månedlig
# -----------------------------------------------------------------------
def print_section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print("=" * 70)


def annualise(r: float, days: int) -> float:
    return (1 + r) ** (365.25 / max(days, 1)) - 1


# Equity-kurve normaliseret til start-af-2026
if len(equity_2026) >= 2:
    eq_norm = equity_2026 / equity_2026.iloc[0]
    strat_total = float(eq_norm.iloc[-1] - 1)
    n_days = (equity_2026.index[-1] - equity_2026.index[0]).days
    strat_cagr = annualise(strat_total, n_days)
    strat_daily = equity_2026.pct_change().dropna()
    strat_sharpe = (
        float(strat_daily.mean() / strat_daily.std() * np.sqrt(252))
        if strat_daily.std() > 0
        else float("nan")
    )
    strat_dd = float((eq_norm / eq_norm.cummax() - 1).min())
else:
    strat_total = strat_cagr = strat_sharpe = strat_dd = float("nan")

spy_norm = spy_close / spy_close.iloc[0]
spy_total = float(spy_norm.iloc[-1] - 1)
spy_n_days = (spy_close.index[-1] - spy_close.index[0]).days
spy_cagr = annualise(spy_total, spy_n_days)
spy_sharpe = (
    float(spy_ret.mean() / spy_ret.std() * np.sqrt(252))
    if spy_ret.std() > 0
    else float("nan")
)
spy_dd = float((spy_norm / spy_norm.cummax() - 1).min())

print_section(f"OVERORDNET 2026 (pr. {END})")
print(f"{'Metric':<25} {'ML Ranker':>12} {'SPY':>12}")
print("-" * 50)
print(f"{'Totalafkast':<25} {strat_total:>11.2%} {spy_total:>11.2%}")
print(f"{'Annualiseret (CAGR)':<25} {strat_cagr:>11.2%} {spy_cagr:>11.2%}")
print(f"{'Sharpe (2026)':<25} {strat_sharpe:>11.2f} {spy_sharpe:>11.2f}")
print(f"{'Max Drawdown':<25} {strat_dd:>11.2%} {spy_dd:>11.2%}")

# -----------------------------------------------------------------------
# MÅNEDLIG PERFORMANCE
# -----------------------------------------------------------------------
print_section(f"MÅNEDLIGT AFKAST {YEAR}")

if len(equity_2026) >= 2:
    monthly_strat = equity_2026.resample("ME").last().pct_change().dropna()
    monthly_spy = spy_close.resample("ME").last().pct_change().dropna()

    months = sorted(set(monthly_strat.index) | set(monthly_spy.index))
    print(f"{'Måned':<12} {'ML Ranker':>12} {'SPY':>12} {'Forskel':>12}")
    print("-" * 50)
    for m in months:
        s = monthly_strat.get(m, float("nan"))
        b = monthly_spy.get(m, float("nan"))
        diff = s - b if not (np.isnan(s) or np.isnan(b)) else float("nan")
        diff_str = f"{diff:>+11.2%}" if not np.isnan(diff) else f"{'—':>12}"
        print(f"{m.strftime('%Y-%m'):<12} {s:>11.2%} {b:>11.2%} {diff_str}")

# -----------------------------------------------------------------------
# UGENTLIG PERFORMANCE (holdings perioder)
# -----------------------------------------------------------------------
print_section(f"UGENTLIG AFKAST (rebalanceringsperioder) {YEAR}")

if not holdings_2026.empty:
    print(f"{'Uge':<12} {'Afkast':>10} {'SPY uge':>10} {'Alfa':>10}  Aktier holdt")
    print("-" * 75)
    for _, row in holdings_2026.iterrows():
        pdate = row["date"]
        ret = row["period_return"]
        # Find SPY-afkast for samme uge
        spy_slice = spy_ret[spy_ret.index >= pd.Timestamp(pdate)]
        spy_slice = spy_slice[
            spy_slice.index < pd.Timestamp(pdate) + pd.Timedelta(days=7)
        ]
        spy_week = (
            float((1 + spy_slice).prod() - 1) if len(spy_slice) > 0 else float("nan")
        )
        alfa = ret - spy_week if not np.isnan(spy_week) else float("nan")
        alfa_str = f"{alfa:>+9.2%}" if not np.isnan(alfa) else f"{'—':>10}"
        n = row["n_held"]
        print(
            f"{pdate.strftime('%Y-%m-%d'):<12} {ret:>9.2%} {spy_week:>9.2%} {alfa_str}  ({n} aktier)"
        )

# -----------------------------------------------------------------------
# HANDELSOVERSIGT: hvad er købt/solgt
# -----------------------------------------------------------------------
print_section(f"HANDELSOVERSIGT {YEAR} — KØBT / SOLGT PER UGE")

if len(holdings_2026) >= 2:
    prev_set: set[str] = set()
    # Find sidst holdede fra 2025 for at se første uges salg
    prev_2025 = holdings[holdings["date"].dt.year < YEAR]
    if not prev_2025.empty:
        prev_set = set(prev_2025.iloc[-1]["tickers"])
        print(f"\nINITIAL PORTEFØLJE ind i {YEAR} ({len(prev_set)} aktier):")
        print("  " + ", ".join(sorted(prev_set)))

    total_buys = 0
    total_sells = 0
    for _, row in holdings_2026.iterrows():
        curr_set = set(row["tickers"])
        bought = sorted(curr_set - prev_set)
        sold = sorted(prev_set - curr_set)
        total_buys += len(bought)
        total_sells += len(sold)
        if bought or sold:
            print(
                f"\n{row['date'].strftime('%Y-%m-%d')}  (afkast: {row['period_return']:+.2%})"
            )
            if bought:
                print(f"  KØB  ({len(bought)}): {', '.join(bought)}")
            if sold:
                print(f"  SALG ({len(sold)}): {', '.join(sold)}")
        else:
            print(f"{row['date'].strftime('%Y-%m-%d')}  — ingen ændringer")
        prev_set = curr_set

    n_periods = len(holdings_2026)
    print(
        f"\nTotal: {total_buys} køb og {total_sells} salg over {n_periods} perioder "
        f"({total_buys / n_periods:.1f} køb/periode i gennemsnit)"
    )

# -----------------------------------------------------------------------
# VINDERE OG TABERE — per aktie akkumuleret i 2026
# -----------------------------------------------------------------------
print_section(f"VINDERE OG TABERE PER AKTIE {YEAR}")

ticker_stats: dict[str, dict] = {}
if len(holdings_2026) >= 2:
    prev_set = set()
    prev_2025 = holdings[holdings["date"].dt.year < YEAR]
    if not prev_2025.empty:
        prev_set = set(prev_2025.iloc[-1]["tickers"])

    rows = holdings_2026.to_dict("records")
    for i, row in enumerate(rows):
        curr_set = set(row["tickers"])
        period_ret = row["period_return"]
        # Antag equal-weight — hver aktie bidrager period_ret / n
        n = row["n_held"]
        per_ticker = period_ret / n if n > 0 else 0.0
        for t in curr_set:
            if t not in ticker_stats:
                ticker_stats[t] = {"periods": 0, "total_ret": 0.0, "wins": 0}
            ticker_stats[t]["periods"] += 1
            ticker_stats[t]["total_ret"] += per_ticker
            if per_ticker > 0:
                ticker_stats[t]["wins"] += 1
        prev_set = curr_set

# Lav dataframe og sorter
stats_df = pd.DataFrame(
    [{"ticker": t, **v} for t, v in ticker_stats.items()]
).sort_values("total_ret", ascending=False)

if not stats_df.empty:
    stats_df["win_rate"] = stats_df["wins"] / stats_df["periods"]
    print("\nTOP 10 VINDERE:")
    print(f"{'Ticker':<10} {'Uger holdt':>10} {'Bidrag':>10} {'Win%':>8}")
    print("-" * 42)
    for _, r in stats_df.head(10).iterrows():
        print(
            f"{r['ticker']:<10} {int(r['periods']):>10} {r['total_ret']:>9.2%} {r['win_rate']:>7.0%}"
        )

    print("\nBUND 10 TABERE:")
    print(f"{'Ticker':<10} {'Uger holdt':>10} {'Bidrag':>10} {'Win%':>8}")
    print("-" * 42)
    for _, r in stats_df.tail(10).sort_values("total_ret").iterrows():
        print(
            f"{r['ticker']:<10} {int(r['periods']):>10} {r['total_ret']:>9.2%} {r['win_rate']:>7.0%}"
        )

# -----------------------------------------------------------------------
# KONSISTENS-CHECK: win rate, batting average osv.
# -----------------------------------------------------------------------
print_section(f"KVALITETSMÅL {YEAR}")

if not holdings_2026.empty:
    rets = holdings_2026["period_return"].values
    wins = (rets > 0).sum()
    losses = (rets <= 0).sum()
    avg_win = rets[rets > 0].mean() if wins > 0 else 0.0
    avg_loss = rets[rets <= 0].mean() if losses > 0 else 0.0
    profit_factor = (
        abs(rets[rets > 0].sum() / rets[rets <= 0].sum())
        if losses > 0 and rets[rets <= 0].sum() != 0
        else float("nan")
    )
    expectancy = rets.mean()

    # SPY ugentlig
    spy_weekly = spy_ret.resample("W").apply(lambda x: (1 + x).prod() - 1).dropna()
    spy_weekly_2026 = spy_weekly[spy_weekly.index.year >= YEAR]
    spy_wins = (spy_weekly_2026 > 0).sum()
    spy_wr = (
        spy_wins / len(spy_weekly_2026) if len(spy_weekly_2026) > 0 else float("nan")
    )

    print(f"{'Mål':<30} {'ML Ranker':>12} {'SPY (uger)':>12}")
    print("-" * 56)
    print(f"{'Perioder':<30} {len(rets):>12} {len(spy_weekly_2026):>12}")
    print(f"{'Win rate':<30} {wins / len(rets):>11.1%} {spy_wr:>11.1%}")
    print(f"{'Gns. gevinst (vinderuge)':<30} {avg_win:>11.2%}")
    print(f"{'Gns. tab (taber-uge)':<30} {avg_loss:>11.2%}")
    print(f"{'Profit factor':<30} {profit_factor:>11.2f}")
    print(f"{'Forventet afkast per periode':<30} {expectancy:>11.2%}")
    print(f"{'Bedste uge':<30} {rets.max():>11.2%}")
    print(f"{'Dårligste uge':<30} {rets.min():>11.2%}")

    # Konsekutive tab
    streak = max_lose_streak = cur_streak = 0
    for r in rets:
        if r <= 0:
            cur_streak += 1
            max_lose_streak = max(max_lose_streak, cur_streak)
        else:
            cur_streak = 0
    print(f"{'Længste taber-streak':<30} {max_lose_streak:>11} uger")

    # Alpha vs SPY per uge
    if len(equity_2026) >= 2:
        alpha_annualised = strat_cagr - spy_cagr
        print(f"\n{'Alfa vs SPY (annualiseret)':<30} {alpha_annualised:>+11.2%}")

# -----------------------------------------------------------------------
# FX-JUSTERING: USD/DKK kursrisiko + vekselgebyr
# -----------------------------------------------------------------------
print_section(f"FX-JUSTERING FOR DANSK INVESTOR (USD/DKK) {YEAR}")

FX_CONVERSION_FEE = 0.005  # Saxo Classic: 0.5% per konvertering (én vej)

try:
    usddkk_raw = yf.download(
        "USDDKK=X",
        start=f"{YEAR}-01-01",
        end=str(END),
        auto_adjust=True,
        progress=False,
    )
    usddkk = usddkk_raw["Close"].squeeze().dropna()
    usddkk.index = pd.to_datetime(usddkk.index)

    if len(usddkk) < 2:
        raise ValueError("For lidt USDDKK data")

    # USD/DKK normaliseret til årets start
    fx_start = usddkk.iloc[0]
    fx_end = usddkk.iloc[-1]
    fx_change = fx_end / fx_start - 1

    print(f"\nUSD/DKK ved årsstart:  {fx_start:.4f}")
    print(f"USD/DKK nu:            {fx_end:.4f}")
    print(f"USD/DKK ændring {YEAR}: {fx_change:>+.2%}  ", end="")
    if fx_change < 0:
        print("(USD svaekket - dansk investor taber paa valuta)")
    else:
        print("(USD styrket - dansk investor vinder paa valuta)")

    # Juster equity-kurve med daglig USD/DKK
    if len(equity_2026) >= 2:
        # Reindex FX til equity-datoer (ffill weekends/huller)
        fx_aligned = usddkk.reindex(equity_2026.index, method="ffill")
        fx_norm = fx_aligned / fx_aligned.iloc[0]

        eq_dkk = equity_2026 / equity_2026.iloc[0] * fx_norm

        # Beregn FX-konverteringsomkostning
        # Tæl nye køb i 2026 (unikke positioner der åbnes)
        n_new_buys = 0
        n_new_sells = 0
        prev_fx_set: set[str] = set()
        prev_2025_fx = holdings[holdings["date"].dt.year < YEAR]
        if not prev_2025_fx.empty:
            prev_fx_set = set(prev_2025_fx.iloc[-1]["tickers"])
        for _, row in holdings_2026.iterrows():
            curr = set(row["tickers"])
            n_new_buys += len(curr - prev_fx_set)
            n_new_sells += len(prev_fx_set - curr)
            prev_fx_set = curr
        # FX-gebyr: 0.5% per handel × vægten af positionen (1/top_n)
        top_n_fx = holdings_2026["n_held"].iloc[0] if not holdings_2026.empty else 15
        fx_drag_pct = (n_new_buys + n_new_sells) * FX_CONVERSION_FEE / top_n_fx

        dkk_total = float(eq_dkk.iloc[-1] - 1) - fx_drag_pct
        dkk_cagr = annualise(dkk_total, n_days)

        print(f"\nHandler i {YEAR}: {n_new_buys} kob + {n_new_sells} salg")
        print(
            f"FX-konverteringsgebyr (0.5% x {n_new_buys + n_new_sells} handler / {top_n_fx} positioner): {fx_drag_pct:>+.2%}"
        )
        print("-" * 65)
        print(f"{'Totalafkast':<35} {strat_total:>13.2%} {dkk_total:>13.2%}")
        print(f"{'Annualiseret (CAGR)':<35} {strat_cagr:>13.2%} {dkk_cagr:>13.2%}")
        print(f"{'USD/DKK bidrag':<35} {'—':>14} {fx_change:>+13.2%}")
        print(f"{'FX-gebyr bidrag':<35} {'—':>14} {-fx_drag_pct:>+13.2%}")

        # Månedlig DKK vs USD
        print(
            f"\n{'Måned':<12} {'USD afkast':>12} {'DKK afkast':>12} {'FX bidrag':>12}"
        )
        print("-" * 50)
        monthly_usd = equity_2026.resample("ME").last().pct_change().dropna()
        monthly_dkk = eq_dkk.resample("ME").last().pct_change().dropna()
        monthly_fx = usddkk.resample("ME").last().pct_change().dropna()
        all_months = sorted(set(monthly_usd.index) | set(monthly_dkk.index))
        for m in all_months:
            u = monthly_usd.get(m, float("nan"))
            d = monthly_dkk.get(m, float("nan"))
            f = monthly_fx.get(m, float("nan"))
            print(f"{m.strftime('%Y-%m'):<12} {u:>11.2%} {d:>11.2%} {f:>+11.2%}")

except Exception as e:
    print(f"Kunne ikke hente USDDKK data: {e}")

print()

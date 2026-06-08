"""pages/3_Daily_Breakout.py

Dashboard-side: Daily Breakout screener.

Viser de aktier med hoejest historisk sandsynlighed for at stige
>= TARGET% i dag, baseret paa gaardagens lukkekurser.

Data hentes fra disk-cache (opdateret af run_daily.py).
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "research"))

from app_cache import get_service, get_universe
from strategies.daily_breakout import (
    TARGET_RETURN,
    DKK_PER_USD,
    TARGET_PROFIT_DKK,
    DailyBreakoutStrategy,
)

st.set_page_config(
    page_title="Daily Breakout · ScanStrat", page_icon="⚡", layout="wide"
)
st.title("⚡ Daily Breakout")
st.caption(
    f"Aktier med hoejest P(+{TARGET_RETURN:.0%} i dag) baseret pa gaardagens data"
)

# ---------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------
st.sidebar.header("Parametre")

region = st.sidebar.selectbox(
    "Univers",
    [
        "All",
        "All US",
        "All EU",
        "US (S&P 500)",
        "NASDAQ-100",
        "Small Cap",
        "EU Small Cap",
    ],
    index=0,
)
region_map = {
    "All": "All",
    "All US": "All US",
    "All EU": "All EU",
    "US (S&P 500)": "US",
    "NASDAQ-100": "NASDAQ-100",
    "Small Cap": "Small Cap",
    "EU Small Cap": "EU Small Cap",
}

top_n = st.sidebar.slider("Vis top N kandidater", 1, 20, 5)
min_prob = st.sidebar.slider(
    f"Min. sandsynlighed for +{TARGET_RETURN:.0%}", 0, 100, 20, step=5
)

st.sidebar.divider()
profit_dkk = st.sidebar.number_input(
    "Profitmal (DKK)",
    min_value=500,
    max_value=20_000,
    value=int(TARGET_PROFIT_DKK),
    step=500,
)
dkk_per_usd = st.sidebar.number_input(
    "DKK/USD kurs",
    min_value=4.0,
    max_value=12.0,
    value=float(DKK_PER_USD),
    step=0.1,
    format="%.1f",
)


# ---------------------------------------------------------------
# Data og screener
# ---------------------------------------------------------------
@st.cache_data(ttl=1800, show_spinner="Korer Daily Breakout screener...")
def run_screener(universe_key: str, as_of: date) -> pd.DataFrame:
    service = get_service()
    universe = get_universe(region_map[universe_key])
    end = as_of
    start = end - timedelta(days=400)
    data = service.get_batch(universe, start, end)

    strategy = DailyBreakoutStrategy()
    return strategy.rank(data, as_of=as_of)


as_of = date.today() - timedelta(days=1)  # brug gaardagens lukkekurser
# spring weekender over
while as_of.weekday() >= 5:
    as_of -= timedelta(days=1)

with st.spinner(
    f"Screener kores for {get_universe(region_map[region]).__len__()} aktier..."
):
    try:
        ranked = run_screener(region, as_of)
    except Exception as exc:
        st.error(f"Fejl: {exc}")
        st.stop()

if ranked.empty:
    st.warning("Ingen aktier passerede screener-kriterierne i dag.")
    st.stop()

# ---------------------------------------------------------------
# Filtrering og visning
# ---------------------------------------------------------------
ranked = ranked[ranked["prob_breakout"] >= min_prob / 100].head(top_n)

if ranked.empty:
    st.warning(f"Ingen aktier med sandsynlighed >= {min_prob}%.")
    st.stop()

st.success(f"**{len(ranked)} kandidater** per {as_of} (top-{top_n}, min {min_prob}%)")

# Beregn position-stoerrelse og forventet profit
position_usd = profit_dkk / TARGET_RETURN / dkk_per_usd
ranked = ranked.copy()
ranked["Sandsynlighed"] = (ranked["prob_breakout"] * 100).round(1).astype(str) + "%"
ranked["Position (USD)"] = f"${position_usd:,.0f}"
ranked["Position (DKK)"] = f"{position_usd * dkk_per_usd:,.0f}"
ranked["Forventet profit (DKK)"] = (
    (ranked["prob_breakout"] * TARGET_RETURN * position_usd * dkk_per_usd)
    .round(0)
    .astype(int)
)

display_cols = [
    "rank",
    "ticker",
    "Sandsynlighed",
    "Position (USD)",
    "Forventet profit (DKK)",
]
st.dataframe(
    ranked[display_cols].set_index("rank"),
    use_container_width=True,
)

st.caption(
    f"Position-stoerrelse beregnet som: {profit_dkk} DKK / {TARGET_RETURN:.0%} / {dkk_per_usd} DKK per USD"
    f" = ${position_usd:,.0f} pr. aktie. "
    f"'Forventet profit' = sandsynlighed x profitmal."
)

# ---------------------------------------------------------------
# Opdateringsstatus
# ---------------------------------------------------------------
st.divider()
st.markdown("**Cache-status**")
service = get_service()
universe_list = get_universe(region_map[region])
fetch_start = as_of - timedelta(days=400)
n_cached = service.count_cached(universe_list, fetch_start, as_of)
n_total = len(universe_list)
st.progress(n_cached / n_total if n_total else 0)
st.caption(
    f"{n_cached}/{n_total} tickers i cache. "
    "Koed `uv run python research/run_daily.py` for at opdatere."
)

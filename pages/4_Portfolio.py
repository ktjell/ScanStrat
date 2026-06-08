"""pages/4_Portfolio.py

Dashboard: morgenrapport for din personlige portefoelge.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.data import CacheManager, DataService, YFinanceLoader
from engine.config import Settings
from portfolio.holdings import HOLDINGS
from portfolio.signals import compute_signals, Signal
from portfolio.charts import plot_holding

st.set_page_config(page_title="Portfolio · ScanStrat", page_icon="💼", layout="wide")
st.title("💼 Min Portefoelge")

# ---------------------------------------------------------------
# Farver / ikoner
# ---------------------------------------------------------------
_COLOR = {"SAELG": "🔴", "ADVARSEL": "🟡", "BEHOLD": "🟢", "INGEN DATA": "⚫"}
_BG = {
    "SAELG": "#ffe5e5",
    "ADVARSEL": "#fff8e0",
    "BEHOLD": "#e8f5e9",
    "INGEN DATA": "#f5f5f5",
}


# ---------------------------------------------------------------
# Data
# ---------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="Henter kursdata...")
def load_signals(as_of: date) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    settings = Settings.default()
    service = DataService(YFinanceLoader(), CacheManager(settings.cache), settings)
    tickers = [h.ticker for h in HOLDINGS]
    end = as_of
    start = end - timedelta(days=400)
    data = service.get_batch(tickers, start, end)
    signals = compute_signals(data, HOLDINGS, as_of=as_of)
    return [
        {
            "action": s.action,
            "ticker": s.ticker,
            "name": s.name,
            "close": s.close,
            "sma50": s.sma50,
            "sma200": s.sma200,
            "rsi": s.rsi,
            "return_4w": s.return_4w,
            "dist_52w_high": s.dist_52w_high,
            "reasons": s.reasons,
        }
        for s in signals
    ], data


today = date.today()
signals_raw, price_data = load_signals(today)
signals = [type("S", (), r)() for r in signals_raw]  # dict -> simple obj for compat

st.caption(f"Opdateret: {today}  |  {len(HOLDINGS)} aktier i portefoelgen")

# ---------------------------------------------------------------
# Overblik
# ---------------------------------------------------------------
n_sell = sum(1 for s in signals_raw if s["action"] == "SAELG")
n_warn = sum(1 for s in signals_raw if s["action"] == "ADVARSEL")
n_hold = sum(1 for s in signals_raw if s["action"] == "BEHOLD")

col1, col2, col3 = st.columns(3)
col1.metric("🔴 Saelg", n_sell)
col2.metric("🟡 Advarsel", n_warn)
col3.metric("🟢 Behold", n_hold)

st.divider()

# ---------------------------------------------------------------
# Tabel med alle aktier
# ---------------------------------------------------------------
rows = []
for s in signals_raw:
    icon = _COLOR.get(s["action"], "⚫")
    rows.append(
        {
            "Signal": f"{icon} {s['action']}",
            "Ticker": s["ticker"],
            "Navn": s["name"],
            "Kurs": round(s["close"], 2) if not pd.isna(s["close"]) else None,
            "SMA50": round(s["sma50"], 2) if not pd.isna(s["sma50"]) else None,
            "SMA200": round(s["sma200"], 2) if not pd.isna(s["sma200"]) else None,
            "RSI": round(s["rsi"], 1) if not pd.isna(s["rsi"]) else None,
            "4-ugers afkast": f"{s['return_4w']:.1%}"
            if not pd.isna(s["return_4w"])
            else None,
            "Afstand 52u-hoj": f"{s['dist_52w_high']:.1%}"
            if not pd.isna(s["dist_52w_high"])
            else None,
            "Signaler": " · ".join(s["reasons"]),
        }
    )

df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------
# Detalje per aktie
# ---------------------------------------------------------------
st.divider()
st.subheader("Detaljer")

for s in signals_raw:
    icon = _COLOR.get(s["action"], "⚫")
    with st.expander(
        f"{icon} {s['ticker']}  —  {s['name']}  ({s['action']})",
        expanded=(s["action"] in ("SAELG", "ADVARSEL")),
    ):
        for r in s["reasons"]:
            if s["action"] == "SAELG":
                st.error(r)
            elif s["action"] == "ADVARSEL":
                st.warning(r)
            else:
                st.success(r)

        if not pd.isna(s["close"]):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Kurs", f"{s['close']:.2f}")
            c2.metric(
                "SMA50",
                f"{s['sma50']:.2f}" if not pd.isna(s["sma50"]) else "N/A",
                delta=f"{s['close'] / s['sma50'] - 1:.1%}"
                if not pd.isna(s["sma50"])
                else None,
            )
            c3.metric("RSI", f"{s['rsi']:.1f}" if not pd.isna(s["rsi"]) else "N/A")
            c4.metric(
                "4-ugers afkast",
                f"{s['return_4w']:.1%}" if not pd.isna(s["return_4w"]) else "N/A",
            )

        # Chart
        df = price_data.get(s["ticker"])
        if df is not None and not df.empty:
            sig_obj = Signal(
                ticker=s["ticker"],
                name=s["name"],
                action=s["action"],
                reasons=s["reasons"],
                close=s["close"],
                sma50=s["sma50"],
                sma200=s["sma200"],
                rsi=s["rsi"],
                return_4w=s["return_4w"],
                dist_52w_high=s["dist_52w_high"],
            )
            fig = plot_holding(df, sig_obj)
            st.pyplot(fig)

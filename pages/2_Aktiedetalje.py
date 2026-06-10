"""Page 2 — Aktiedetalje: kurs, SMA, RSI og feature-snapshot."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

from app_cache import get_service
from features.feature_engine import FeatureEngine

st.set_page_config(page_title="Aktiedetalje · ScanStrat", page_icon="📊", layout="wide")

# Tilbage-knap hvis vi kom fra ML Live
if st.session_state.get("came_from_ml_live"):
    if st.button("← Tilbage til ML Live"):
        st.session_state["came_from_ml_live"] = False
        st.switch_page("pages/5_ML_Live.py")

st.title("📊 Aktiedetalje")

# ---------------------------------------------------------------
# Ticker-valg
# ---------------------------------------------------------------
default_ticker = st.session_state.get("selected_ticker", "AAPL")
ticker = st.text_input("Ticker", value=default_ticker).upper().strip()
lookback_years = st.sidebar.slider("Historik (år)", 1, 5, 2)

if not ticker:
    st.stop()


# ---------------------------------------------------------------
# Data
# ---------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="Henter kursdata…")
def load_ticker(ticker: str, years: int) -> pd.DataFrame:
    service = get_service()
    end = date.today()
    start = end - timedelta(days=365 * years)
    return service.get(ticker, start, end)


df = load_ticker(ticker, lookback_years)

if df.empty:
    st.error(f"Ingen data for **{ticker}**. Tjek at tickeren er korrekt.")
    st.stop()


# Selskabsbeskrivelse
@st.cache_data(ttl=86400, show_spinner=False)
def load_company_info(t: str) -> dict:
    try:
        return yf.Ticker(t).info
    except Exception:
        return {}


info = load_company_info(ticker)
name = info.get("longName") or info.get("shortName", ticker)
summary = info.get("longBusinessSummary")
with st.expander(f"ℹ️ {name}", expanded=False):
    if summary:
        st.write(summary)
    else:
        st.write("Ingen beskrivelse tilgængelig.")

close = df["close"]
sma50 = close.rolling(50).mean()
sma200 = close.rolling(200).mean()

# RSI
delta = close.diff()
alpha = 1.0 / 14
gain = delta.clip(lower=0).ewm(alpha=alpha, adjust=False, min_periods=14).mean()
loss = (-delta).clip(lower=0).ewm(alpha=alpha, adjust=False, min_periods=14).mean()
rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

# ---------------------------------------------------------------
# Plot: Kurs + SMAs + RSI (Plotly)
# ---------------------------------------------------------------
fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    row_heights=[0.7, 0.3],
    vertical_spacing=0.05,
    subplot_titles=(f"{ticker} — Kurs", "RSI 14"),
)

# Panel 1: Kurs
fig.add_trace(
    go.Scatter(
        x=df.index, y=close, name="Close", line=dict(color="steelblue", width=1.5)
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=sma50,
        name="SMA 50",
        line=dict(color="orange", width=1, dash="dash"),
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=sma200,
        name="SMA 200",
        line=dict(color="red", width=1, dash="dash"),
    ),
    row=1,
    col=1,
)

# Death/golden cross baggrund
if len(sma200.dropna()) > 0:
    cross_signal = (sma50 < sma200).astype(float)
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=np.where(cross_signal == 1, close.max() * 1.05, np.nan),
            fill="tozeroy",
            fillcolor="rgba(255,0,0,0.05)",
            line=dict(width=0),
            showlegend=False,
            name="Death cross zone",
        ),
        row=1,
        col=1,
    )

# Panel 2: RSI
fig.add_trace(
    go.Scatter(x=df.index, y=rsi, name="RSI 14", line=dict(color="purple", width=1)),
    row=2,
    col=1,
)
fig.add_hline(y=70, line=dict(color="red", dash="dash", width=0.8), row=2, col=1)
fig.add_hline(y=30, line=dict(color="green", dash="dash", width=0.8), row=2, col=1)
fig.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.05, line_width=0, row=2, col=1)
fig.add_hrect(y0=0, y1=30, fillcolor="green", opacity=0.05, line_width=0, row=2, col=1)

fig.update_yaxes(title_text="Pris (USD)", row=1, col=1)
fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
fig.update_layout(
    height=550, margin=dict(t=40, b=20), legend=dict(orientation="h", y=1.02)
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------
# KØB / HOLD / SÆLG anbefaling
# ---------------------------------------------------------------
st.subheader("📋 Samlet anbefaling")

_RANKING_FILE = Path(__file__).parent.parent / "live" / "data" / "latest_ranking.json"

# Hent ML-score hvis tilgængelig
ml_score: float | None = None
ml_rank: int | None = None
try:
    if _RANKING_FILE.exists():
        ranking = json.loads(_RANKING_FILE.read_text(encoding="utf-8"))
        entry = next(
            (s for s in ranking.get("top_stocks", []) if s["ticker"] == ticker), None
        )
        if entry:
            ml_score = entry["ml_score"]
            ml_rank = entry["rank"]
except Exception:
    pass

# Beregn aktuelle værdier
latest_close = float(close.iloc[-1])
latest_sma50 = float(sma50.iloc[-1]) if not np.isnan(sma50.iloc[-1]) else None
latest_sma200 = float(sma200.iloc[-1]) if not np.isnan(sma200.iloc[-1]) else None
latest_rsi = float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else None

# RSI-trend (stigende eller faldende seneste 5 dage)
rsi_clean = rsi.dropna()
rsi_trend = (
    float(rsi_clean.iloc[-1] - rsi_clean.iloc[-6]) if len(rsi_clean) >= 6 else 0.0
)

# ---------------------------------------------------------------
# Pointsystem: hver signal giver +1 (bullish), -1 (bearish), 0 (neutral)
# ---------------------------------------------------------------
signals: list[tuple[str, int, str]] = []  # (navn, point, beskrivelse)

# 1. ML-score
if ml_score is not None:
    if ml_score >= 0.60:
        signals.append(
            (
                "ML Ranker",
                2,
                f"Top-{ml_rank} · score {ml_score * 100:.1f}% — stærkt bullish",
            )
        )
    elif ml_score >= 0.52:
        signals.append(
            (
                "ML Ranker",
                1,
                f"Top-{ml_rank} · score {ml_score * 100:.1f}% — svagt bullish",
            )
        )
    elif ml_score >= 0.48:
        signals.append(("ML Ranker", 0, f"Score {ml_score * 100:.1f}% — neutral"))
    else:
        signals.append(
            ("ML Ranker", -1, f"Score {ml_score * 100:.1f}% — under gennemsnit")
        )
else:
    signals.append(("ML Ranker", 0, "Ikke i top-30 (ingen score)"))

# 2. Kurs vs SMA50
if latest_sma50:
    pct50 = (latest_close / latest_sma50 - 1) * 100
    if pct50 > 3:
        signals.append(("Kurs vs SMA50", 1, f"{pct50:+.1f}% over — bullish momentum"))
    elif pct50 > -3:
        signals.append(("Kurs vs SMA50", 0, f"{pct50:+.1f}% — tæt på SMA50"))
    else:
        signals.append(("Kurs vs SMA50", -1, f"{pct50:+.1f}% under — svagt"))

# 3. Golden / Death Cross
if latest_sma50 and latest_sma200:
    if latest_sma50 > latest_sma200:
        signals.append(("Golden/Death Cross", 1, "Golden Cross — SMA50 over SMA200"))
    else:
        signals.append(("Golden/Death Cross", -1, "Death Cross — SMA50 under SMA200"))

# 4. RSI
if latest_rsi is not None:
    if latest_rsi < 30:
        signals.append(
            ("RSI 14", 1, f"RSI {latest_rsi:.0f} — oversolgt (potentielt bounce)")
        )
    elif latest_rsi < 45:
        signals.append(("RSI 14", 0, f"RSI {latest_rsi:.0f} — svag zone"))
    elif latest_rsi <= 70:
        signals.append(("RSI 14", 1, f"RSI {latest_rsi:.0f} — sund zone"))
    else:
        signals.append(("RSI 14", -1, f"RSI {latest_rsi:.0f} — overkøbt"))

# 5. RSI-trend
if rsi_trend > 5:
    signals.append(("RSI-trend", 1, f"Stiger +{rsi_trend:.0f}pt (5d) — momentum op"))
elif rsi_trend < -5:
    signals.append(("RSI-trend", -1, f"Falder {rsi_trend:.0f}pt (5d) — momentum ned"))
else:
    signals.append(("RSI-trend", 0, f"Stabil ({rsi_trend:+.0f}pt, 5d)"))

# 6. Kurs vs SMA200 (langsigtet trend)
if latest_sma200:
    pct200 = (latest_close / latest_sma200 - 1) * 100
    if pct200 > 5:
        signals.append(("Langsigtet trend", 1, f"{pct200:+.1f}% over SMA200 — bullish"))
    elif pct200 > -5:
        signals.append(("Langsigtet trend", 0, f"{pct200:+.1f}% vs SMA200 — neutral"))
    else:
        signals.append(
            ("Langsigtet trend", -1, f"{pct200:+.1f}% under SMA200 — bearish")
        )

# ---------------------------------------------------------------
# Samlet score
# ---------------------------------------------------------------
total = sum(p for _, p, _ in signals)
max_score = sum(
    2 if n == "ML Ranker" else 1 for n, _, _ in signals
)  # ML tæller dobbelt
min_score = -max_score

# Normaliser til -100..+100
score_pct = int(total / max_score * 100) if max_score else 0

if score_pct >= 40:
    verdict = "KØB"
    verdict_color = "#1b5e20"
    verdict_bg = "#e8f5e9"
    verdict_icon = "🟢"
elif score_pct >= 10:
    verdict = "SVAGT KØB"
    verdict_color = "#2e7d32"
    verdict_bg = "#f1f8e9"
    verdict_icon = "🟢"
elif score_pct >= -10:
    verdict = "HOLD"
    verdict_color = "#e65100"
    verdict_bg = "#fff8e1"
    verdict_icon = "🟡"
elif score_pct >= -40:
    verdict = "SVAGT SÆLG"
    verdict_color = "#b71c1c"
    verdict_bg = "#fce4ec"
    verdict_icon = "🔴"
else:
    verdict = "SÆLG"
    verdict_color = "#b71c1c"
    verdict_bg = "#fde8e8"
    verdict_icon = "🔴"

# Vis samlet anbefaling
st.markdown(
    f"<div style='background:{verdict_bg};padding:16px 20px;border-radius:10px;"
    f"border-left:5px solid {verdict_color};margin-bottom:12px'>"
    f"<span style='font-size:1.6rem;font-weight:bold;color:{verdict_color}'>"
    f"{verdict_icon} {verdict}</span>"
    f"<span style='font-size:1rem;color:#555;margin-left:16px'>"
    f"Signal score: {score_pct:+d}/100</span>"
    f"</div>",
    unsafe_allow_html=True,
)

# Signal-tabel
sig_rows = []
for name, pts, desc in signals:
    if pts > 0:
        icon = "🟢" * pts
    elif pts < 0:
        icon = "🔴" * abs(pts)
    else:
        icon = "🟡"
    sig_rows.append({"Signal": name, "": icon, "Beskrivelse": desc})

st.dataframe(
    pd.DataFrame(sig_rows),
    use_container_width=True,
    hide_index=True,
    column_config={
        "Signal": st.column_config.TextColumn(width="medium"),
        "": st.column_config.TextColumn(width="small"),
        "Beskrivelse": st.column_config.TextColumn(width="large"),
    },
)

st.caption(
    "⚠️ Anbefaling er baseret på tekniske indikatorer + ML-model. "
    "Ikke finansiel rådgivning — brug som ét input blandt flere."
)

st.divider()

# ---------------------------------------------------------------
# Feature snapshot
# ---------------------------------------------------------------
st.subheader("Feature snapshot")

engine = FeatureEngine.default()
row = engine.compute_row(ticker, df)

feature_rows = []
for k, v in row.items():
    if k == "ticker":
        continue
    if isinstance(v, float) and np.isnan(v):
        val_str = "—"
    elif isinstance(v, float):
        val_str = f"{v:.4f}"
    else:
        val_str = str(v)
    feature_rows.append({"Feature": k, "Værdi": val_str})

feat_df = pd.DataFrame(feature_rows).set_index("Feature")
st.dataframe(feat_df, use_container_width=False)

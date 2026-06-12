"""pages/5_ML_Live.py

Live ML Ranker dashboard:
  - Top-15 aktier med ML-score (P(slaar SPY naeste uge))
  - Portefoelge tracker: nuvaerende beholdning vs. anbefalet
  - Opdateres automatisk fra live/data/latest_ranking.json
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

_ROOT = Path(__file__).parent.parent

import sys

sys.path.insert(0, str(_ROOT))
from live.macro_indicators import get_all_indicators, overall_signal

st.set_page_config(
    page_title="ML Live · ScanStrat",
    page_icon="🤖",
    layout="wide",
)

# ------------------------------------------------------------------
# Stier
# ------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent
_RANKING_FILE = _ROOT / "live" / "data" / "latest_ranking.json"
_PORTFOLIO_FILE = _ROOT / "live" / "data" / "portfolio.json"
_PAPER_FILE = _ROOT / "live" / "data" / "paper_trades.json"

# ------------------------------------------------------------------
# Paper trade helpers
# ------------------------------------------------------------------


def _load_paper_trades() -> dict:
    if not _PAPER_FILE.exists():
        return {"positions": {}, "closed_trades": [], "equity_history": []}
    try:
        return json.loads(_PAPER_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"positions": {}, "closed_trades": [], "equity_history": []}


# ------------------------------------------------------------------
# Portfolio helpers
# ------------------------------------------------------------------
# Format: [{"ticker": "AAPL", "buy_date": "2026-01-01", "buy_price": 150.0}, ...]
# Bagudkompatibelt: gamle string-lister migreres automatisk.


def _load_portfolio() -> list[dict]:
    if not _PORTFOLIO_FILE.exists():
        return []
    try:
        raw = json.loads(_PORTFOLIO_FILE.read_text(encoding="utf-8"))
        # Migrer gammelt format (liste af strings)
        if raw and isinstance(raw[0], str):
            today = str(date.today())
            return [
                {"ticker": t.upper(), "buy_date": today, "buy_price": None} for t in raw
            ]
        return raw
    except Exception:
        return []


def _save_portfolio(positions: list[dict]) -> None:
    _PORTFOLIO_FILE.parent.mkdir(parents=True, exist_ok=True)
    _PORTFOLIO_FILE.write_text(
        json.dumps(
            sorted(positions, key=lambda p: p["ticker"]), indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_prices(tickers: tuple[str, ...]) -> dict[str, float]:
    """Hent seneste intradag-kurs for en tuple af tickers (5 min cache).
    Bruger 5-minutters intervaller så kursen afspejler hvad der sker i løbet af dagen.
    Falder tilbage til daglig close hvis intradag-data ikke er tilgængeligt.
    """
    if not tickers:
        return {}
    try:
        raw = yf.download(
            list(tickers), period="1d", interval="5m", auto_adjust=True, progress=False
        )
        if raw.empty:
            raise ValueError("Ingen intradag-data")
        close = raw["Close"] if "Close" in raw.columns else raw
        if isinstance(close, pd.Series):
            close = close.to_frame(name=tickers[0])
        last = close.ffill().iloc[-1]
        prices = {
            str(t): float(last[t])
            for t in tickers
            if t in last.index and not pd.isna(last[t])
        }
        # Fallback til daglig close for tickers der mangler intradag-data
        missing = [t for t in tickers if t not in prices]
        if missing:
            raw_daily = yf.download(
                missing, period="2d", auto_adjust=True, progress=False
            )
            if not raw_daily.empty:
                close_d = (
                    raw_daily["Close"] if "Close" in raw_daily.columns else raw_daily
                )
                if isinstance(close_d, pd.Series):
                    close_d = close_d.to_frame(name=missing[0])
                last_d = close_d.ffill().iloc[-1]
                for t in missing:
                    if t in last_d.index and not pd.isna(last_d[t]):
                        prices[str(t)] = float(last_d[t])
        return prices
    except Exception:
        return {}


def _get_buy_price(ticker: str) -> float | None:
    """Hent dagens close (bruges som købskurs ved tilføjelse)."""
    prices = _fetch_prices((ticker,))
    return prices.get(ticker)


# ------------------------------------------------------------------
# Ranking helpers
# ------------------------------------------------------------------


def _load_ranking() -> dict | None:
    if not _RANKING_FILE.exists():
        return None
    try:
        return json.loads(_RANKING_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None


def _age_str(iso: str | None) -> str:
    if not iso:
        return "ukendt"
    try:
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            # Ingen timezone info — antag lokal tid
            now = datetime.now()
        else:
            now = datetime.now(tz=dt.tzinfo)
        diff = now - dt
        m = int(diff.total_seconds() // 60)
        if m < 0:
            m = 0
        if m < 60:
            return f"{m} min siden"
        return f"{m // 60}t {m % 60}m siden"
    except Exception:
        return iso


# ------------------------------------------------------------------
# Side
# ------------------------------------------------------------------

st.title("🤖 ML Ranker — Live")

# Auto-refresh hvert 10. minut
st.markdown(
    '<meta http-equiv="refresh" content="600">',
    unsafe_allow_html=True,
)

ranking = _load_ranking()

if ranking is None:
    st.error(
        "Ingen ranking-data fundet. Kør `uv run python live/scorer.py` for at generere data."
    )
    st.stop()

updated_at = ranking.get("updated_at")
as_of = ranking.get("as_of_date", "?")
model_trained = ranking.get("model_trained_at")
universe_size = ranking.get("universe_size", "?")
top_stocks: list[dict] = ranking.get("top_stocks", [])
previous_top15: set[str] = set(ranking.get("previous_top15", []))

# ------------------------------------------------------------------
# Header-info
# ------------------------------------------------------------------
col_a, col_b, col_c = st.columns(3)
col_a.metric("Sidst opdateret", _age_str(updated_at))
col_b.metric("Data pr.", as_of)
col_c.metric("Univers", f"{universe_size} aktier")

st.caption(f"Model traenet: {_age_str(model_trained)}")
st.divider()

# ------------------------------------------------------------------
# Makro-indikatorer
# ------------------------------------------------------------------
st.subheader("🌍 Makro-regime (4 indikatorer)")

_STATUS_ICON = {"GROEN": "🟢", "GOEL": "🟡", "ROED": "🔴", "GRAA": "⚫"}
_STATUS_COLOR = {
    "GROEN": "#e8f5e9",
    "GOEL": "#fff8e0",
    "ROED": "#fde8e8",
    "GRAA": "#f5f5f5",
}


@st.cache_data(ttl=3600, show_spinner="Henter makrodata...")
def load_macro() -> tuple[list[dict], str, str]:
    inds = get_all_indicators()
    status, msg = overall_signal(inds)
    return inds, status, msg


@st.cache_data(ttl=3600, show_spinner="Beregner tekniske indikatorer...")
def _fetch_technicals(tickers: tuple[str, ...]) -> dict[str, dict]:
    """
    Beregn SMA50, SMA200, golden/death cross og pris vs SMA50 for en liste af tickers.
    Returnerer dict keyed på ticker.
    """
    if not tickers:
        return {}
    try:
        raw = yf.download(
            list(tickers), period="300d", auto_adjust=True, progress=False
        )
        if raw.empty:
            return {}
        close = raw["Close"] if "Close" in raw.columns else raw
        if isinstance(close, pd.Series):
            close = close.to_frame(name=tickers[0])

        result = {}
        for ticker in tickers:
            if ticker not in close.columns:
                continue
            s = close[ticker].dropna()
            if len(s) < 50:
                continue
            price = float(s.iloc[-1])
            sma50 = float(s.tail(50).mean())
            sma200 = float(s.tail(200).mean()) if len(s) >= 200 else None

            above_sma50 = price > sma50
            pct_vs_sma50 = (price / sma50 - 1) * 100

            if sma200 is not None:
                # Golden cross: SMA50 > SMA200 (og var under for 5 dage siden = frisk kryds)
                sma50_prev = float(s.iloc[-6:-1].mean()) if len(s) >= 6 else sma50
                sma200_prev = (
                    float(s.tail(205).head(200).mean()) if len(s) >= 205 else sma200
                )
                golden = sma50 > sma200
                fresh_cross = (sma50 > sma200) and (sma50_prev <= sma200_prev)
                death_cross = sma50 < sma200
                fresh_death = (sma50 < sma200) and (sma50_prev >= sma200_prev)

                if fresh_cross:
                    cross_str = "🌟 Frisk GC"
                elif golden:
                    cross_str = "🟡 Golden Cross"
                elif fresh_death:
                    cross_str = "💀 Frisk DC"
                elif death_cross:
                    cross_str = "🔻 Death Cross"
                else:
                    cross_str = "—"
            else:
                cross_str = "< 200d data"

            # RSI 14
            delta = s.diff()
            alpha = 1.0 / 14
            gain = (
                delta.clip(lower=0)
                .ewm(alpha=alpha, adjust=False, min_periods=14)
                .mean()
            )
            loss = (
                (-delta)
                .clip(lower=0)
                .ewm(alpha=alpha, adjust=False, min_periods=14)
                .mean()
            )
            rsi_s = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
            rsi_clean = rsi_s.dropna()
            rsi_val = float(rsi_clean.iloc[-1]) if len(rsi_clean) >= 1 else None
            rsi_trend = (
                float(rsi_clean.iloc[-1] - rsi_clean.iloc[-6])
                if len(rsi_clean) >= 6
                else 0.0
            )

            result[ticker] = {
                "price": price,
                "sma50": round(sma50, 2),
                "sma200": round(sma200, 2) if sma200 else None,
                "above_sma50": above_sma50,
                "pct_vs_sma50": round(pct_vs_sma50, 1),
                "cross": cross_str,
                "golden": golden if sma200 else None,
                "rsi": round(rsi_val, 1) if rsi_val else None,
                "rsi_trend": round(rsi_trend, 1),
            }
        return result
    except Exception:
        return {}


def _compute_verdict(ml_score: float | None, t: dict) -> tuple[str, str]:
    """
    Beregn KØB/HOLD/SÆLG baseret på ML-score + tekniske indikatorer.
    Returnerer (verdict_str, farve-emoji).
    """
    points = 0
    max_pts = 0

    # ML-score (tæller dobbelt)
    if ml_score is not None:
        max_pts += 2
        if ml_score >= 0.60:
            points += 2
        elif ml_score >= 0.52:
            points += 1
        elif ml_score < 0.48:
            points -= 1

    # Kurs vs SMA50
    if t.get("pct_vs_sma50") is not None:
        max_pts += 1
        pct50 = t["pct_vs_sma50"]
        if pct50 > 3:
            points += 1
        elif pct50 < -3:
            points -= 1

    # Golden/Death cross
    if t.get("golden") is not None:
        max_pts += 1
        points += 1 if t["golden"] else -1

    # RSI
    rsi = t.get("rsi")
    if rsi is not None:
        max_pts += 1
        if rsi < 30:
            points += 1  # oversolgt = bounce-potential
        elif rsi <= 70:
            points += 1
        else:
            points -= 1  # overkøbt

    # RSI-trend
    rsi_trend = t.get("rsi_trend", 0)
    if abs(rsi_trend) > 5:
        max_pts += 1
        points += 1 if rsi_trend > 0 else -1

    # Kurs vs SMA200
    if t.get("sma200") and t.get("price"):
        pct200 = (t["price"] / t["sma200"] - 1) * 100
        max_pts += 1
        if pct200 > 5:
            points += 1
        elif pct200 < -5:
            points -= 1

    if max_pts == 0:
        return "—", "⚪"

    score_pct = int(points / max_pts * 100)
    if score_pct >= 40:
        return "KØB", "🟢"
    elif score_pct >= 10:
        return "Svagt KØB", "🟢"
    elif score_pct >= -10:
        return "HOLD", "🟡"
    elif score_pct >= -40:
        return "Svagt SÆLG", "🔴"
    else:
        return "SÆLG", "🔴"


indicators, macro_status, macro_msg = load_macro()

# Samlet signal
overall_icon = _STATUS_ICON[macro_status]
overall_bg = _STATUS_COLOR[macro_status]
st.markdown(
    f"<div style='background:{overall_bg};padding:12px;border-radius:8px;"
    f"font-size:1.1rem;font-weight:bold'>"
    f"{overall_icon} {macro_msg}</div>",
    unsafe_allow_html=True,
)
st.write("")

# 4 indikatorer i kolonner
cols = st.columns(4)
for col, ind in zip(cols, indicators):
    icon = _STATUS_ICON[ind["status"]]
    bg = _STATUS_COLOR[ind["status"]]
    with col:
        st.markdown(
            f"<div style='background:{bg};padding:10px;border-radius:8px;min-height:90px'>"
            f"<b>{icon} {ind['name']}</b><br>"
            f"<span style='font-size:0.9rem'>{ind['signal']}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

with st.expander("ℹ️ Hvad betyder indikatorerne?"):
    for ind in indicators:
        icon = _STATUS_ICON[ind["status"]]
        st.markdown(f"**{icon} {ind['name']}**: {ind['description']}")
    st.markdown("""
**Samlet signal:**
- 🟢 Alle grønne → 100% investeret som normalt
- 🟡 1-2 røde/gule → reducer positioner, hold mere cash
- 🔴 3-4 røde → gå defensiv (cash, guld, obligationer)
""")

st.divider()

# ------------------------------------------------------------------
# Top-15 ranking
# ------------------------------------------------------------------
st.subheader("📊 Top-15 anbefalinger")
st.caption("P(slår SPY næste uge) — jo højere, jo bedre")

portfolio: list[dict] = _load_portfolio()
portfolio_set = {p["ticker"].upper() for p in portfolio}
recommended_set = {s["ticker"] for s in top_stocks[:15]}
# Buffer: sælg kun hvis aktie er faldet ud af top-20
top20_set = {s["ticker"] for s in top_stocks[:20]}

# Ny i top-15 siden sidst (ikke i forrige top-15)
new_in_top15 = recommended_set - previous_top15
# Udgået fra top-15 siden sidst (var i forrige, ikke i nuværende)
exited_top15 = previous_top15 - recommended_set

if top_stocks:
    # Hent tekniske indikatorer for top-15 (1t cache)
    top15_tickers = tuple(s["ticker"] for s in top_stocks[:15])
    tech = _fetch_technicals(top15_tickers)
    paper = _load_paper_trades()
    paper_positions = paper.get("positions", {})

    rows = []
    for s in top_stocks[:15]:
        ticker = s["ticker"]
        is_new = ticker in new_in_top15
        in_port = ticker in portfolio_set

        t = tech.get(ticker, {})
        sma50_str = (
            f"{'🟢' if t.get('above_sma50') else '🔴'} {t['pct_vs_sma50']:+.1f}%"
            if t
            else "—"
        )
        cross_str = t.get("cross", "—") if t else "—"

        verdict, v_icon = _compute_verdict(s["ml_score"], t)

        # Paper afkast siden debut på top-15
        paper_str = "—"
        pos = paper_positions.get(ticker)
        if pos and pos.get("entry_price") and t.get("price"):
            paper_ret = (t["price"] / pos["entry_price"] - 1) * 100
            arrow = "🟢" if paper_ret >= 0 else "🔴"
            paper_str = f"{arrow} {paper_ret:+.1f}%"
        elif pos and pos.get("entry_date"):
            paper_str = f"📅 {pos['entry_date']}"

        rows.append(
            {
                "#": s["rank"],
                "Aktie": ticker,
                "Score": f"{s['ml_score_pct']:.1f}%",
                "Ny": "🆕" if is_new else "",
                "✅": "✅" if in_port else "",
                "Paper %": paper_str,
                "Vs SMA50": sma50_str,
                "GC / DC": cross_str,
                "Anbefaling": f"{v_icon} {verdict}",
            }
        )
    df15 = pd.DataFrame(rows)
    st.dataframe(
        df15,
        use_container_width=True,
        hide_index=True,
        column_config={
            "#": st.column_config.NumberColumn(width="small"),
            "Aktie": st.column_config.TextColumn(width="small"),
            "Score": st.column_config.TextColumn(width="small"),
            "Ny": st.column_config.TextColumn(width="small"),
            "✅": st.column_config.TextColumn(width="small"),
            "Paper %": st.column_config.TextColumn(width="small"),
            "Vs SMA50": st.column_config.TextColumn(width="medium"),
            "GC / DC": st.column_config.TextColumn(width="medium"),
            "Anbefaling": st.column_config.TextColumn(width="medium"),
        },
    )

    # Sammenfatning af ændringer
    if new_in_top15 or exited_top15:
        c1, c2 = st.columns(2)
        if new_in_top15:
            c1.success(f"**🆕 Nye i top-15:** {', '.join(sorted(new_in_top15))}")
        if exited_top15:
            c2.warning(f"**⬇️ Faldet ud af top-15:** {', '.join(sorted(exited_top15))}")
    else:
        st.caption("_Ingen ændringer i top-15 siden sidst._")

    # Genvej til aktiedetalje
    st.markdown("**🔍 Åbn aktiedetalje:**")
    btn_cols = st.columns(8)
    for i, s in enumerate(top_stocks[:15]):
        if btn_cols[i % 8].button(
            s["ticker"], key=f"detail_{s['ticker']}", use_container_width=True
        ):
            st.session_state["selected_ticker"] = s["ticker"]
            st.session_state["came_from_ml_live"] = True
            st.switch_page("pages/2_Aktiedetalje.py")

else:
    st.warning("Ingen ranking-data tilgaengelig.")

st.divider()

# ------------------------------------------------------------------
# Paper Portfolio P&L
# ------------------------------------------------------------------
st.subheader("📈 Paper Portfolio P&L")
st.caption("Simuleret equal-weight portefølje der følger top-15 listen automatisk")

_paper = _load_paper_trades()
_paper_positions = _paper.get("positions", {})
_closed_trades = _paper.get("closed_trades", [])
_equity_history = _paper.get("equity_history", [])

if not _paper_positions and not _closed_trades:
    st.info(
        "Paper trading starter automatisk næste gang scorer.py kører (kl. 8, 12, 18 eller 22). Kør manuelt for at starte nu: `python live/scorer.py`"
    )
else:
    # Åbne positioner med live P&L
    if _paper_positions:
        top15_tickers_paper = tuple(s["ticker"] for s in top_stocks[:15])
        _paper_prices = _fetch_prices(top15_tickers_paper)

        open_rows = []
        open_returns = []
        for ticker, pos in sorted(_paper_positions.items()):
            entry_price = pos.get("entry_price")
            current_price = _paper_prices.get(ticker)
            if entry_price and current_price:
                ret_pct = (current_price / entry_price - 1) * 100
                open_returns.append(ret_pct)
                arrow = "🟢" if ret_pct >= 0 else "🔴"
                ret_str = f"{arrow} {ret_pct:+.1f}%"
            else:
                ret_str = "—"
                ret_pct = None
            open_rows.append(
                {
                    "Aktie": ticker,
                    "Debut": pos.get("entry_date", "?"),
                    "Købt @": f"${entry_price:.2f}" if entry_price else "—",
                    "Kurs nu": f"${current_price:.2f}" if current_price else "—",
                    "Afkast": ret_str,
                }
            )

        # Samlet P&L for åbne positioner
        if open_returns:
            avg_open = sum(open_returns) / len(open_returns)
            c1, c2, c3 = st.columns(3)
            c1.metric("Åbne positioner", len(open_returns))
            c2.metric("Snit afkast (åbne)", f"{avg_open:+.1f}%")
            best = max(open_returns)
            worst = min(open_returns)
            c3.metric("Bedst / Dårligst", f"{best:+.1f}% / {worst:+.1f}%")

        st.dataframe(pd.DataFrame(open_rows), use_container_width=True, hide_index=True)

    # Lukkede handler
    if _closed_trades:
        closed_returns = [
            t["return_pct"] for t in _closed_trades if t.get("return_pct") is not None
        ]
        if closed_returns:
            avg_closed = sum(closed_returns) / len(closed_returns)
            wins = sum(1 for r in closed_returns if r > 0)
            st.caption(
                f"**Lukkede handler:** {len(closed_returns)} | Snit: {avg_closed:+.1f}% | Win rate: {wins}/{len(closed_returns)} ({wins / len(closed_returns) * 100:.0f}%)"
            )

        with st.expander(f"📋 Historik ({len(_closed_trades)} lukkede handler)"):
            closed_rows = []
            for t in sorted(
                _closed_trades, key=lambda x: x.get("exit_date", ""), reverse=True
            ):
                ret = t.get("return_pct")
                arrow = ("🟢" if ret >= 0 else "🔴") if ret is not None else "—"
                closed_rows.append(
                    {
                        "Aktie": t["ticker"],
                        "Købt": t.get("entry_date", "?"),
                        "Solgt": t.get("exit_date", "?"),
                        "Købt @": f"${t['entry_price']:.2f}"
                        if t.get("entry_price")
                        else "—",
                        "Solgt @": f"${t['exit_price']:.2f}"
                        if t.get("exit_price")
                        else "—",
                        "Afkast": f"{arrow} {ret:+.1f}%" if ret is not None else "—",
                    }
                )
            st.dataframe(
                pd.DataFrame(closed_rows), use_container_width=True, hide_index=True
            )

st.divider()

# ------------------------------------------------------------------
# Handlinger: KØB og SÆLG
# ------------------------------------------------------------------
st.subheader("🔔 Handlinger")

# KØB: i top-15 men ikke i portefølje
to_buy = recommended_set - portfolio_set
# SÆLG: i portefølje OG faldet ud af top-20 (buffer)
to_sell = portfolio_set - top20_set

col_buy, col_sell = st.columns(2)

with col_buy:
    if to_buy:
        st.markdown("### 🟢 Køb")
        for ticker in sorted(to_buy):
            entry = next((s for s in top_stocks if s["ticker"] == ticker), None)
            score_str = f"{entry['ml_score_pct']:.1f}%" if entry else "?"
            rank_str = str(entry["rank"]) if entry else "?"
            badge = "🆕" if ticker in new_in_top15 else ""
            st.markdown(
                f"<div style='background:#e8f5e9;padding:8px 12px;border-radius:6px;"
                f"margin-bottom:6px'><b>{badge} {ticker}</b> &nbsp; "
                f"#{rank_str} · Score: {score_str}</div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown("### 🟢 Køb")
        st.caption("_Ingen nye køb — portefølje matcher top-15._")

with col_sell:
    if to_sell:
        st.markdown("### 🔴 Sælg")
        st.caption("_Aktier der er faldet ud af top-20 (buffer på 5 pladser)_")
        for ticker in sorted(to_sell):
            entry = next((s for s in top_stocks if s["ticker"] == ticker), None)
            if entry:
                score_str = f"{entry['ml_score_pct']:.1f}% (rank #{entry['rank']})"
            else:
                score_str = "ikke i top-30"
            st.markdown(
                f"<div style='background:#fde8e8;padding:8px 12px;border-radius:6px;"
                f"margin-bottom:6px'><b>{ticker}</b> &nbsp; {score_str}</div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown("### 🔴 Sælg")
        st.caption("_Ingen salg — alle beholdninger er i top-20._")

# Aktier der stadig er i top-15 men nær grænsen (rang 13-15)
near_exit = [s for s in top_stocks[12:15] if s["ticker"] in portfolio_set]
if near_exit:
    tickers_str = ", ".join(f"{s['ticker']} (#{s['rank']})" for s in near_exit)
    st.caption(f"⚠️ Hold øje med: {tickers_str} — lavt placeret i top-15")

st.divider()

# ------------------------------------------------------------------
# Portefølje tracker
# ------------------------------------------------------------------
st.subheader("💼 Min portefølje")

if portfolio:
    # Hent aktuelle kurser (5 min cache)
    all_tickers = tuple(sorted(portfolio_set))
    current_prices = _fetch_prices(all_tickers)
    today = date.today()

    rows = []
    total_invested = 0.0
    total_now = 0.0

    for pos in sorted(portfolio, key=lambda p: p["ticker"]):
        ticker = pos["ticker"]
        buy_date_str = pos.get("buy_date") or "—"
        buy_price = pos.get("buy_price")
        curr_price = current_prices.get(ticker)

        # Beregn P&L
        shares = pos.get("shares")
        if buy_price and curr_price:
            pnl_pct = (curr_price / buy_price - 1) * 100
            pnl_emoji = "🟢" if pnl_pct >= 0 else "🔴"
            if shares:
                pnl_usd = (curr_price - buy_price) * shares
                pnl_str = f"{pnl_emoji} {pnl_pct:+.1f}% (${pnl_usd:+,.0f})"
            else:
                pnl_str = f"{pnl_emoji} {pnl_pct:+.1f}%"
            total_invested += (buy_price * shares) if shares else buy_price
            total_now += (curr_price * shares) if shares else curr_price
        elif curr_price and not buy_price:
            pnl_str = "—"
            pnl_emoji = "⚪"
        else:
            pnl_str = "—"
            pnl_emoji = "⚪"

        # Dage holdt
        if buy_date_str != "—":
            try:
                days_held = (today - date.fromisoformat(buy_date_str)).days
                days_str = f"{days_held}d"
            except Exception:
                days_str = "—"
        else:
            days_str = "—"

        # ML status
        in_top15 = ticker in recommended_set
        in_top20 = ticker in top20_set
        ml_entry = next((s for s in top_stocks if s["ticker"] == ticker), None)
        rank_str = str(ml_entry["rank"]) if ml_entry else "—"
        if in_top15:
            ml_status = "✅ Behold"
        elif in_top20:
            ml_status = "🟡 Observér"
        else:
            ml_status = "🔴 Sælg"

        rows.append(
            {
                "Ticker": ticker,
                "Købt": buy_date_str,
                "Antal": f"{shares:.2f}" if shares else "—",
                "Købt kurs": f"${buy_price:.2f}" if buy_price else "—",
                "Kurs nu": f"${curr_price:.2f}" if curr_price else "—",
                "Afkast": pnl_str,
                "Dage": days_str,
                "ML #": rank_str,
                "Status": ml_status,
            }
        )

    port_df = pd.DataFrame(rows)
    st.dataframe(
        port_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Ticker": st.column_config.TextColumn(width="small"),
            "Købt": st.column_config.TextColumn(width="small"),
            "Antal": st.column_config.TextColumn(width="small"),
            "Købt kurs": st.column_config.TextColumn(width="small"),
            "Kurs nu": st.column_config.TextColumn(width="small"),
            "Afkast": st.column_config.TextColumn(width="medium"),
            "Dage": st.column_config.TextColumn(width="small"),
            "ML #": st.column_config.TextColumn(width="small"),
            "Status": st.column_config.TextColumn(width="medium"),
        },
    )

    # Samlet P&L hvis vi har nok data
    positions_with_prices = [
        p for p in portfolio if p.get("buy_price") and current_prices.get(p["ticker"])
    ]
    if positions_with_prices:
        total_buy = sum(p["buy_price"] for p in positions_with_prices)
        total_cur = sum(current_prices[p["ticker"]] for p in positions_with_prices)
        overall_pct = (total_cur / total_buy - 1) * 100
        n = len(positions_with_prices)
        col_tot1, col_tot2, col_tot3 = st.columns(3)
        col_tot1.metric("Positioner med kursdata", f"{n}/{len(portfolio)}")
        col_tot2.metric(
            "Gns. urealiseret afkast", f"{overall_pct:+.1f}%", delta_color="normal"
        )
        best = max(
            positions_with_prices,
            key=lambda p: current_prices[p["ticker"]] / p["buy_price"],
        )
        worst = min(
            positions_with_prices,
            key=lambda p: current_prices[p["ticker"]] / p["buy_price"],
        )
        col_tot3.metric(
            "Bedst / Dårligst",
            f"{best['ticker']} / {worst['ticker']}",
        )
else:
    st.info("Din portefølje er tom. Tilføj aktier nedenfor.")

st.divider()

# ------------------------------------------------------------------
# Rediger portefølje
# ------------------------------------------------------------------
with st.expander("✏️ Rediger portefølje", expanded=not bool(portfolio)):
    st.caption("Tilføj/fjern aktier. Købskurs hentes automatisk (dagens kurs).")

    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    col1.markdown("<small>Ticker</small>", unsafe_allow_html=True)
    col2.markdown("<small>Købt kurs ($)</small>", unsafe_allow_html=True)
    col3.markdown("<small>Antal aktier</small>", unsafe_allow_html=True)
    col4.markdown("<small>&nbsp;</small>", unsafe_allow_html=True)
    new_ticker = (
        col1.text_input(
            "Tilføj ticker", placeholder="fx AAPL", label_visibility="collapsed"
        )
        .strip()
        .upper()
    )
    custom_price = col2.number_input(
        "Købt til (valgfrit)",
        min_value=0.0,
        value=0.0,
        format="%.2f",
        label_visibility="collapsed",
        help="Angiv købt kurs — ellers bruges dagens kurs",
    )
    custom_shares = col3.number_input(
        "Antal aktier",
        min_value=0.0,
        value=0.0,
        format="%.4f",
        label_visibility="collapsed",
        help="Antal aktier du har købt",
    )
    if col4.button("Tilføj", use_container_width=True) and new_ticker:
        if new_ticker not in portfolio_set:
            buy_price = custom_price if custom_price > 0 else _get_buy_price(new_ticker)
            portfolio.append(
                {
                    "ticker": new_ticker,
                    "buy_date": str(date.today()),
                    "buy_price": round(buy_price, 4) if buy_price else None,
                    "shares": round(custom_shares, 4) if custom_shares > 0 else None,
                }
            )
            _save_portfolio(portfolio)
            price_str = f" (kurs: ${buy_price:.2f})" if buy_price else ""
            st.success(f"{new_ticker} tilføjet{price_str}")
            st.rerun()

    if portfolio:
        st.write("**Fjern aktie:**")
        btn_cols = st.columns(5)
        for i, pos in enumerate(sorted(portfolio, key=lambda p: p["ticker"])):
            ticker = pos["ticker"]
            if btn_cols[i % 5].button(
                f"🗑 {ticker}", key=f"del_{ticker}", use_container_width=True
            ):
                portfolio = [p for p in portfolio if p["ticker"] != ticker]
                _save_portfolio(portfolio)
                st.rerun()

    if st.button(
        "📋 Synkroniser med ML top-15",
        help="Tilføj manglende top-15 aktier og behold eksisterende",
    ):
        prices = _fetch_prices(tuple(sorted(recommended_set - portfolio_set)))
        for ticker in sorted(recommended_set - portfolio_set):
            buy_price = prices.get(ticker)
            portfolio.append(
                {
                    "ticker": ticker,
                    "buy_date": str(date.today()),
                    "buy_price": round(buy_price, 4) if buy_price else None,
                }
            )
        # Fjern aktier der er faldet ud af top-20
        portfolio = [
            p
            for p in portfolio
            if p["ticker"] in top20_set or p["ticker"] in portfolio_set
        ]
        _save_portfolio(portfolio)
        st.success("Portefølje synkroniseret med ML top-15")
        st.rerun()

st.divider()

# ------------------------------------------------------------------
# Positionsstørrelse-beregner
# ------------------------------------------------------------------
st.subheader("🧮 Positionsstørrelse-beregner")
st.caption(
    "Vælg aktier fra top-15, angiv vægte og investeringsbeløb — beregneren viser hvor meget du skal sætte i hver."
)

top15_options = [s["ticker"] for s in top_stocks[:15]]
default_selected = [
    s["ticker"] for s in top_stocks[:15] if s["ticker"] in portfolio_set
] or top15_options[:5]

col_sel, col_amt = st.columns([3, 1])
with col_sel:
    selected = st.multiselect(
        "Vælg aktier",
        options=top15_options,
        default=default_selected,
        format_func=lambda t: (
            f"{t}  (#{next(s['rank'] for s in top_stocks if s['ticker'] == t)} · {next(s['ml_score_pct'] for s in top_stocks if s['ticker'] == t):.1f}%)"
        ),
    )
with col_amt:
    currency = st.selectbox(
        "Valuta", ["USD", "DKK"], index=1, label_visibility="visible"
    )
    total_amount = st.number_input(
        f"Samlet beløb ({currency})",
        min_value=0.0,
        value=50_000.0 if currency == "DKK" else 10_000.0,
        step=1000.0,
        format="%.0f",
    )

if selected and total_amount > 0:
    st.markdown(
        "**Vægte per aktie** (træk i slider — normaliseres automatisk til 100%)"
    )

    weights: dict[str, float] = {}
    n = len(selected)

    # Brug ML-score som default-vægt (høj score = større andel)
    score_map = {s["ticker"]: s["ml_score"] for s in top_stocks}
    raw_weights = {t: score_map.get(t, 0.5) for t in selected}
    weight_sum = sum(raw_weights.values())

    slider_cols = st.columns(min(n, 5))
    for i, ticker in enumerate(selected):
        default_w = round(raw_weights[ticker] / weight_sum * 100, 1)
        with slider_cols[i % 5]:
            weights[ticker] = st.slider(
                ticker,
                min_value=0.0,
                max_value=100.0,
                value=default_w,
                step=0.5,
                key=f"w_{ticker}",
            )

    total_w = sum(weights.values())

    if total_w == 0:
        st.warning("Alle vægte er 0 — juster sliderne.")
    else:
        # Hent aktuelle kurser
        prices = _fetch_prices(tuple(selected))

        # Byg resultat-tabel
        rows = []
        for ticker in selected:
            w_norm = weights[ticker] / total_w  # normaliseret vægt
            amount = total_amount * w_norm
            price = prices.get(ticker)
            shares = amount / price if price else None

            rows.append(
                {
                    "Aktie": ticker,
                    "Vægt": f"{w_norm * 100:.1f}%",
                    f"Beløb ({currency})": f"{amount:,.0f}",
                    "Kurs (USD)": f"${price:.2f}" if price else "—",
                    "Antal aktier": f"{shares:.2f}" if shares else "—",
                }
            )

        result_df = pd.DataFrame(rows)
        st.dataframe(
            result_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Aktie": st.column_config.TextColumn(width="small"),
                "Vægt": st.column_config.TextColumn(width="small"),
                f"Beløb ({currency})": st.column_config.TextColumn(width="medium"),
                "Kurs (USD)": st.column_config.TextColumn(width="small"),
                "Antal aktier": st.column_config.TextColumn(width="small"),
            },
        )

        # Totallinje
        c1, c2, c3 = st.columns(3)
        c1.metric("Aktier valgt", len(selected))
        c2.metric(f"Samlet beløb ({currency})", f"{total_amount:,.0f}")
        allocated_pct = min(total_w / total_w * 100, 100)
        c3.metric("Fordelt", f"{allocated_pct:.0f}%")

        # Vis advarsel hvis kurser mangler (fx weekend/markedet lukket)
        missing_prices = [t for t in selected if t not in prices]
        if missing_prices:
            st.caption(
                f"⚠️ Ingen aktuel kurs for: {', '.join(missing_prices)} — antal aktier kan ikke beregnes."
            )

st.divider()

# ------------------------------------------------------------------
# Fuld ranking (top-30)
# ------------------------------------------------------------------
with st.expander("📋 Fuld ranking (top-30)"):
    if top_stocks:
        full_df = pd.DataFrame(top_stocks)
        full_df["Score"] = full_df["ml_score_pct"].apply(lambda x: f"{x:.1f}%")
        full_df["I portefølje"] = full_df["ticker"].apply(
            lambda t: "✅" if t in portfolio_set else ""
        )
        st.dataframe(
            full_df[["rank", "ticker", "Score", "I portefølje"]].rename(
                columns={"rank": "#", "ticker": "Aktie"}
            ),
            use_container_width=True,
            hide_index=True,
        )

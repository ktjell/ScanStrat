"""pages/7_Ensemble_Live.py

Multi-faktor Ensemble Ranker — Live dashboard.
Tre specialiserede XGBoost-modeller (Momentum, Teknisk, Makro) kombineret.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import yfinance as yf

_ROOT = Path(__file__).parent.parent

import sys

sys.path.insert(0, str(_ROOT))

from live.macro_indicators import get_all_indicators, overall_signal

st.set_page_config(
    page_title="Ensemble Live · ScanStrat",
    page_icon="🧠",
    layout="wide",
)

# ------------------------------------------------------------------
# Stier
# ------------------------------------------------------------------
_RANKING_FILE = _ROOT / "live" / "data" / "ensemble_ranking.json"
_PAPER_FILE = _ROOT / "live" / "data" / "ensemble_paper_trades.json"
_PORTFOLIO_FILE = _ROOT / "live" / "data" / "ensemble_portfolio.json"

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _load_ranking() -> dict | None:
    if not _RANKING_FILE.exists():
        return None
    try:
        raw = _RANKING_FILE.read_bytes().decode("utf-8-sig")
        return json.loads(raw)
    except Exception as e:
        st.warning(f"Kunne ikke læse ensemble_ranking.json: {e}")
        return None


def _load_paper_trades() -> dict:
    if not _PAPER_FILE.exists():
        return {"positions": {}, "closed_trades": [], "equity_history": []}
    try:
        return json.loads(_PAPER_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"positions": {}, "closed_trades": [], "equity_history": []}


def _load_portfolio() -> list[dict]:
    if not _PORTFOLIO_FILE.exists():
        return []
    try:
        raw = json.loads(_PORTFOLIO_FILE.read_text(encoding="utf-8"))
        return raw if isinstance(raw, list) else []
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


def _age_str(iso: str | None) -> str:
    if not iso:
        return "ukendt"
    try:
        dt = datetime.fromisoformat(iso)
        now = datetime.now(tz=dt.tzinfo) if dt.tzinfo else datetime.now()
        m = max(0, int((now - dt).total_seconds() // 60))
        return f"{m} min siden" if m < 60 else f"{m // 60}t {m % 60}m siden"
    except Exception:
        return iso


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_prices(tickers: tuple[str, ...]) -> dict[str, float]:
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
        missing = [t for t in tickers if t not in prices]
        if missing:
            raw_d = yf.download(missing, period="2d", auto_adjust=True, progress=False)
            if not raw_d.empty:
                cd = raw_d["Close"] if "Close" in raw_d.columns else raw_d
                if isinstance(cd, pd.Series):
                    cd = cd.to_frame(name=missing[0])
                ld = cd.ffill().iloc[-1]
                for t in missing:
                    if t in ld.index and not pd.isna(ld[t]):
                        prices[str(t)] = float(ld[t])
        return prices
    except Exception:
        return {}


def _get_buy_price(ticker: str) -> float | None:
    return _fetch_prices((ticker,)).get(ticker)


# ------------------------------------------------------------------
# Side
# ------------------------------------------------------------------

st.title("🧠 Ensemble Ranker — Live")
st.caption(
    "Tre specialiserede modeller: **Momentum** (40%) · **Teknisk** (35%) · **Makro** (25%)"
)

st.markdown('<meta http-equiv="refresh" content="600">', unsafe_allow_html=True)

ranking = _load_ranking()

if ranking is None:
    st.error(
        f"Ingen ensemble-data fundet i `{_RANKING_FILE}`. "
        "Kør `uv run python live/ensemble_scorer.py` for at generere data."
    )
    st.stop()

updated_at = ranking.get("updated_at")
as_of = ranking.get("as_of_date", "?")
model_trained = ranking.get("model_trained_at")
universe_size = ranking.get("universe_size", "?")
top_stocks: list[dict] = ranking.get("top_stocks", [])
previous_top15: set[str] = set(ranking.get("previous_top15", []))
model_weights: dict = ranking.get("model_weights", {})

col_a, col_b, col_c = st.columns(3)
col_a.metric("Sidst opdateret", _age_str(updated_at))
col_b.metric("Data pr.", as_of)
col_c.metric("Univers", f"{universe_size} aktier")
st.caption(
    f"Model trænet: {_age_str(model_trained)}  |  "
    f"Vægte: Momentum {model_weights.get('momentum', 0.40):.0%} · "
    f"Teknisk {model_weights.get('technical', 0.35):.0%} · "
    f"Makro {model_weights.get('macro', 0.25):.0%}"
)

st.divider()

# ------------------------------------------------------------------
# Makro-indikatorer
# ------------------------------------------------------------------
_STATUS_ICON = {"GROEN": "🟢", "GOEL": "🟡", "ROED": "🔴", "GRAA": "⚫"}
_STATUS_COLOR = {
    "GROEN": "#e8f5e9",
    "GOEL": "#fff8e0",
    "ROED": "#fde8e8",
    "GRAA": "#f5f5f5",
}

st.subheader("🌍 Makro-regime")


@st.cache_data(ttl=3600, show_spinner="Henter makrodata...")
def load_macro():
    inds = get_all_indicators()
    status, msg = overall_signal(inds)
    return inds, status, msg


indicators, macro_status, macro_msg = load_macro()
overall_icon = _STATUS_ICON[macro_status]
overall_bg = _STATUS_COLOR[macro_status]
st.markdown(
    f"<div style='background:{overall_bg};padding:10px;border-radius:8px;"
    f"font-size:1.0rem;font-weight:bold'>{overall_icon} {macro_msg}</div>",
    unsafe_allow_html=True,
)
st.write("")

st.divider()

# ------------------------------------------------------------------
# Top-15 ranking med individuelle model-scores
# ------------------------------------------------------------------
st.subheader("📊 Top-15 anbefalinger")
st.caption("Vægtet ensemble-score — og hvad de tre modeller individuelt mener")

portfolio = _load_portfolio()
portfolio_set = {p["ticker"].upper() for p in portfolio}
recommended_set = {s["ticker"] for s in top_stocks[:15]}
top20_set = {s["ticker"] for s in top_stocks[:20]}
new_in_top15 = recommended_set - previous_top15
exited_top15 = previous_top15 - recommended_set

paper = _load_paper_trades()
paper_positions = paper.get("positions", {})

if top_stocks:
    top15_tickers = tuple(s["ticker"] for s in top_stocks[:15])
    prices = _fetch_prices(top15_tickers)

    rows = []
    for s in top_stocks[:15]:
        ticker = s["ticker"]
        price = prices.get(ticker)

        # Paper P&L
        pos = paper_positions.get(ticker)
        if pos and pos.get("entry_price") and price:
            paper_ret = (price / pos["entry_price"] - 1) * 100
            arrow = "🟢" if paper_ret >= 0 else "🔴"
            paper_str = f"{arrow} {paper_ret:+.1f}%"
        else:
            paper_str = "—"

        # Individuelle scores som søjler (0-100%)
        def score_bar(v) -> str:
            if v is None:
                return "—"
            pct = round(v * 100, 1)
            bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
            return f"{bar} {pct:.0f}%"

        rows.append(
            {
                "#": s["rank"],
                "Aktie": ticker,
                "Ny": "🆕" if ticker in new_in_top15 else "",
                "✅": "✅" if ticker in portfolio_set else "",
                "Ensemble": f"{s['ensemble_score_pct']:.1f}%",
                "Momentum": score_bar(s.get("momentum_score")),
                "Teknisk": score_bar(s.get("technical_score")),
                "Makro": score_bar(s.get("macro_score")),
                "Paper %": paper_str,
            }
        )

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
        column_config={
            "#": st.column_config.NumberColumn(width="small"),
            "Aktie": st.column_config.TextColumn(width="small"),
            "Ny": st.column_config.TextColumn(width="small"),
            "✅": st.column_config.TextColumn(width="small"),
            "Ensemble": st.column_config.TextColumn(width="small"),
            "Momentum": st.column_config.TextColumn(width="medium"),
            "Teknisk": st.column_config.TextColumn(width="medium"),
            "Makro": st.column_config.TextColumn(width="medium"),
            "Paper %": st.column_config.TextColumn(width="small"),
        },
    )

    if new_in_top15 or exited_top15:
        c1, c2 = st.columns(2)
        if new_in_top15:
            c1.success(f"**🆕 Nye i top-15:** {', '.join(sorted(new_in_top15))}")
        if exited_top15:
            c2.warning(f"**⬇️ Faldet ud:** {', '.join(sorted(exited_top15))}")

    # Genvej til aktiedetalje
    st.markdown("**🔍 Åbn aktiedetalje:**")
    btn_cols = st.columns(8)
    for i, s in enumerate(top_stocks[:15]):
        if btn_cols[i % 8].button(
            s["ticker"], key=f"ens_detail_{s['ticker']}", use_container_width=True
        ):
            st.session_state["selected_ticker"] = s["ticker"]
            st.session_state["came_from_ml_live"] = True
            st.switch_page("pages/2_Aktiedetalje.py")

st.divider()

# ------------------------------------------------------------------
# Paper Portfolio P&L
# ------------------------------------------------------------------
st.subheader("📈 Paper Portfolio P&L")
st.caption("Automatisk equal-weight portefølje der følger ensemble top-15")

_paper_positions = paper.get("positions", {})
_closed_trades = paper.get("closed_trades", [])

if not _paper_positions and not _closed_trades:
    st.info(
        "Paper trading starter automatisk næste gang ensemble_scorer.py kører. "
        "Kør manuelt: `uv run python live/ensemble_scorer.py`"
    )
else:
    if _paper_positions:
        open_tickers = tuple(_paper_positions.keys())
        open_prices = _fetch_prices(open_tickers)

        open_rows = []
        open_returns = []
        for ticker, pos in sorted(_paper_positions.items()):
            entry_price = pos.get("entry_price")
            current_price = open_prices.get(ticker)
            if entry_price and current_price:
                ret_pct = (current_price / entry_price - 1) * 100
                open_returns.append(ret_pct)
                arrow = "🟢" if ret_pct >= 0 else "🔴"
                ret_str = f"{arrow} {ret_pct:+.1f}%"
            else:
                ret_str = "—"
            open_rows.append(
                {
                    "Aktie": ticker,
                    "Debut": pos.get("entry_date", "?"),
                    "Købt @": f"${entry_price:.2f}" if entry_price else "—",
                    "Kurs nu": f"${current_price:.2f}" if current_price else "—",
                    "Afkast": ret_str,
                }
            )

        if open_returns:
            avg_open = sum(open_returns) / len(open_returns)
            c1, c2, c3 = st.columns(3)
            c1.metric("Åbne positioner", len(open_returns))
            c2.metric("Snit afkast (åbne)", f"{avg_open:+.1f}%")
            c3.metric(
                "Bedst / Dårligst",
                f"{max(open_returns):+.1f}% / {min(open_returns):+.1f}%",
            )

        st.dataframe(pd.DataFrame(open_rows), use_container_width=True, hide_index=True)

    if _closed_trades:
        closed_returns = [
            t["return_pct"] for t in _closed_trades if t.get("return_pct") is not None
        ]
        if closed_returns:
            wins = sum(1 for r in closed_returns if r > 0)
            avg = sum(closed_returns) / len(closed_returns)
            st.caption(
                f"**Lukkede:** {len(closed_returns)} | Snit: {avg:+.1f}% | "
                f"Win rate: {wins}/{len(closed_returns)} ({wins / len(closed_returns) * 100:.0f}%)"
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
# Portefølje tracker
# ------------------------------------------------------------------
st.subheader("💼 Min Ensemble-portefølje (manuel)")

with st.expander("✏️ Rediger portefølje", expanded=not bool(portfolio)):
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    col1.markdown("<small>Ticker</small>", unsafe_allow_html=True)
    col2.markdown("<small>Købt kurs ($)</small>", unsafe_allow_html=True)
    col3.markdown("<small>Antal aktier</small>", unsafe_allow_html=True)
    col4.markdown("<small>&nbsp;</small>", unsafe_allow_html=True)

    new_ticker = (
        col1.text_input(
            "Ticker",
            placeholder="fx AAPL",
            label_visibility="collapsed",
            key="ens_ticker",
        )
        .strip()
        .upper()
    )
    custom_price = col2.number_input(
        "Pris",
        min_value=0.0,
        value=0.0,
        format="%.2f",
        label_visibility="collapsed",
        key="ens_price",
    )
    custom_shares = col3.number_input(
        "Antal",
        min_value=0,
        value=0,
        step=1,
        format="%d",
        label_visibility="collapsed",
        key="ens_shares",
    )

    if col4.button("Tilføj", use_container_width=True, key="ens_add") and new_ticker:
        if new_ticker not in portfolio_set:
            buy_price = custom_price if custom_price > 0 else _get_buy_price(new_ticker)
            portfolio.append(
                {
                    "ticker": new_ticker,
                    "buy_date": str(date.today()),
                    "buy_price": round(buy_price, 4) if buy_price else None,
                    "shares": custom_shares if custom_shares > 0 else None,
                }
            )
            _save_portfolio(portfolio)
            st.success(f"{new_ticker} tilføjet")
            st.rerun()

    if portfolio:
        st.write("**Fjern aktie:**")
        btn_cols = st.columns(5)
        for i, pos in enumerate(sorted(portfolio, key=lambda p: p["ticker"])):
            if btn_cols[i % 5].button(
                f"🗑 {pos['ticker']}",
                key=f"ens_del_{pos['ticker']}",
                use_container_width=True,
            ):
                portfolio = [p for p in portfolio if p["ticker"] != pos["ticker"]]
                _save_portfolio(portfolio)
                st.rerun()

if portfolio:
    all_tickers = tuple(p["ticker"] for p in portfolio)
    current_prices = _fetch_prices(all_tickers)
    port_rows = []
    returns_list = []
    for p in sorted(portfolio, key=lambda x: x["ticker"]):
        ticker = p["ticker"]
        buy_price = p.get("buy_price")
        shares = p.get("shares")
        curr = current_prices.get(ticker)
        if buy_price and curr:
            ret_pct = (curr / buy_price - 1) * 100
            returns_list.append(ret_pct)
            arrow = "🟢" if ret_pct >= 0 else "🔴"
            if shares:
                pnl_usd = (curr - buy_price) * shares
                ret_str = f"{arrow} {ret_pct:+.1f}% (${pnl_usd:+,.0f})"
            else:
                ret_str = f"{arrow} {ret_pct:+.1f}%"
        else:
            ret_str = "—"
        port_rows.append(
            {
                "Ticker": ticker,
                "Antal": f"{shares}" if shares else "—",
                "Købt kurs": f"${buy_price:.2f}" if buy_price else "—",
                "Kurs nu": f"${curr:.2f}" if curr else "—",
                "Afkast": ret_str,
            }
        )
    if returns_list:
        c1, c2 = st.columns(2)
        c1.metric("Positioner", len(portfolio))
        c2.metric("Snit afkast", f"{sum(returns_list) / len(returns_list):+.1f}%")
    st.dataframe(pd.DataFrame(port_rows), use_container_width=True, hide_index=True)
else:
    st.info("Porteføljen er tom. Tilføj aktier ovenfor.")

st.divider()

# ------------------------------------------------------------------
# Positionsstørrelse-beregner
# ------------------------------------------------------------------
st.subheader("🧮 Positionsstørrelse-beregner")

col_sel, col_amt = st.columns([2, 1])
with col_sel:
    calc_options = [s["ticker"] for s in top_stocks[:15]]
    selected = st.multiselect(
        "Vælg aktier fra top-15", options=calc_options, key="ens_calc_sel"
    )
with col_amt:
    currency = st.selectbox("Valuta", ["USD", "DKK"], index=1, key="ens_currency")
    total_amount = st.number_input(
        f"Samlet beløb ({currency})",
        min_value=0.0,
        value=50_000.0 if currency == "DKK" else 10_000.0,
        step=1000.0,
        format="%.0f",
        key="ens_amount",
    )

if selected and total_amount > 0:
    calc_prices = _fetch_prices(tuple(selected))
    calc_rows = []
    for ticker in selected:
        price = calc_prices.get(ticker)
        amount = total_amount / len(selected)
        shares = amount / price if price else None
        calc_rows.append(
            {
                "Aktie": ticker,
                f"Beløb ({currency})": f"{amount:,.0f}",
                "Kurs ($)": f"${price:.2f}" if price else "—",
                "Antal aktier": f"{shares:.1f}" if shares else "—",
            }
        )
    st.dataframe(pd.DataFrame(calc_rows), use_container_width=True, hide_index=True)
    st.metric(f"Samlet ({currency})", f"{total_amount:,.0f}")

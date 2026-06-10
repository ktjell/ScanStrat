"""pages/6_PEAD_Live.py

Post-Earnings Announcement Drift (PEAD) — Live dashboard:
  - Aktier med positiv EPS surprise inden for de seneste dage
  - Kommende earnings i universet
  - Paper trade tracker (automatisk entry/exit)
  - Manuel handel: tilføj/fjern egne positioner
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

st.set_page_config(
    page_title="PEAD Live · ScanStrat",
    page_icon="📣",
    layout="wide",
)

# ------------------------------------------------------------------
# Stier
# ------------------------------------------------------------------
_RANKING_FILE = _ROOT / "live" / "data" / "pead_ranking.json"
_PAPER_FILE = _ROOT / "live" / "data" / "pead_paper_trades.json"
_PORTFOLIO_FILE = _ROOT / "live" / "data" / "pead_portfolio.json"

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _load_ranking() -> dict | None:
    if not _RANKING_FILE.exists():
        return None
    try:
        return json.loads(_RANKING_FILE.read_text(encoding="utf-8"))
    except Exception:
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
    """Hent seneste intradag-kurs (5 min cache)."""
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


# ------------------------------------------------------------------
# Side
# ------------------------------------------------------------------

st.title("📣 PEAD — Post-Earnings Drift")
st.caption(
    "Aktier der slår earnings-estimater fortsætter typisk med at stige de næste 10-20 dage. "
    "Strategien åbner automatisk paper trades ved positive EPS surprises."
)

# Auto-refresh hvert 10. minut
st.markdown('<meta http-equiv="refresh" content="600">', unsafe_allow_html=True)

ranking = _load_ranking()

if ranking is None:
    st.error(
        "Ingen PEAD-data fundet. Kør `uv run python live/pead_scorer.py` for at generere data."
    )
    st.stop()

updated_at = ranking.get("updated_at")
as_of = ranking.get("as_of_date", "?")
universe_size = ranking.get("universe_size", "?")
surprises: list[dict] = ranking.get("surprises", [])
upcoming: list[dict] = ranking.get("upcoming_earnings", [])
hold_days: int = ranking.get("hold_days", 20)
threshold_pct: float = ranking.get("surprise_threshold_pct", 3.0)

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Sidst opdateret", _age_str(updated_at))
col_b.metric("Data pr.", as_of)
col_c.metric("Univers", f"{universe_size} aktier")
col_d.metric("EPS surprise threshold", f"+{threshold_pct:.0f}%")

st.divider()

# ------------------------------------------------------------------
# Seneste EPS surprises
# ------------------------------------------------------------------
st.subheader("🎯 Seneste positive EPS surprises")
st.caption(
    f"Aktier der har slået EPS-estimat med >{threshold_pct:.0f}% inden for de seneste dage"
)

if surprises:
    tickers_surp = tuple(s["ticker"] for s in surprises)
    prices_surp = _fetch_prices(tickers_surp)

    rows = []
    for s in surprises:
        ticker = s["ticker"]
        price = prices_surp.get(ticker)
        rows.append(
            {
                "Aktie": ticker,
                "Selskab": s.get("company_name", ticker),
                "Earnings dato": s.get("earnings_date", "?"),
                "EPS faktisk": s.get("actual_eps", "?"),
                "EPS estimat": s.get("estimated_eps", "?"),
                "Surprise": f"+{s['surprise_pct']:.1f}%",
                "Kurs nu": f"${price:.2f}" if price else "—",
            }
        )

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Surprise": st.column_config.TextColumn(width="small"),
            "Kurs nu": st.column_config.TextColumn(width="small"),
        },
    )
else:
    st.info(
        f"Ingen aktier med EPS surprise >{threshold_pct:.0f}% inden for de seneste dage. "
        "Prøv at køre `uv run python live/pead_scorer.py` igen tættere på earnings-sæsonen."
    )

st.divider()

# ------------------------------------------------------------------
# Kommende earnings
# ------------------------------------------------------------------
st.subheader("📅 Kommende earnings (næste 7 dage)")

if upcoming:
    up_rows = []
    for u in upcoming:
        estimated = u.get("estimated_eps")
        up_rows.append(
            {
                "Aktie": u.get("ticker", "?"),
                "Selskab": u.get("company_name", "?"),
                "Earnings dato": u.get("earnings_date", "?"),
                "Estimeret EPS": f"{estimated:.2f}"
                if estimated and not pd.isna(estimated)
                else "—",
            }
        )
    st.dataframe(pd.DataFrame(up_rows), use_container_width=True, hide_index=True)
else:
    st.info("Ingen planlagte earnings inden for de næste 7 dage i universet.")

st.divider()

# ------------------------------------------------------------------
# Paper Portfolio P&L
# ------------------------------------------------------------------
st.subheader("📈 Paper Portfolio P&L")
st.caption(
    f"Automatisk paper trading: køb ved positiv EPS surprise, sælg efter {hold_days} handelsdage"
)

paper = _load_paper_trades()
paper_positions: dict = paper.get("positions", {})
closed_trades: list[dict] = paper.get("closed_trades", [])
equity_history: list[dict] = paper.get("equity_history", [])

if not paper_positions and not closed_trades:
    st.info(
        "Paper trading starter automatisk næste gang pead_scorer.py kører. "
        "Kør manuelt for at starte nu: `uv run python live/pead_scorer.py`"
    )
else:
    # --- Åbne positioner ---
    if paper_positions:
        open_tickers = tuple(paper_positions.keys())
        open_prices = _fetch_prices(open_tickers)

        open_rows = []
        open_returns = []
        for ticker, pos in sorted(paper_positions.items()):
            entry_price = pos.get("entry_price")
            current_price = open_prices.get(ticker)
            days_held = pos.get("hold_days_elapsed", "?")

            # Beregn dage holdt
            try:
                entry_d = date.fromisoformat(pos["entry_date"])
                days_held = (date.today() - entry_d).days
            except Exception:
                days_held = "?"

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
                    "Earnings dato": pos.get("earnings_date", "?"),
                    "Surprise": f"+{pos['surprise_pct']:.1f}%"
                    if pos.get("surprise_pct")
                    else "?",
                    "Debut": pos.get("entry_date", "?"),
                    "Dage holdt": f"{days_held}/{hold_days}",
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
            best = max(open_returns)
            worst = min(open_returns)
            c3.metric("Bedst / Dårligst", f"{best:+.1f}% / {worst:+.1f}%")

        st.dataframe(pd.DataFrame(open_rows), use_container_width=True, hide_index=True)

    # --- Lukkede handler ---
    if closed_trades:
        closed_returns = [
            t["return_pct"] for t in closed_trades if t.get("return_pct") is not None
        ]
        if closed_returns:
            avg_closed = sum(closed_returns) / len(closed_returns)
            wins = sum(1 for r in closed_returns if r > 0)
            st.caption(
                f"**Lukkede handler:** {len(closed_returns)} | "
                f"Snit: {avg_closed:+.1f}% | "
                f"Win rate: {wins}/{len(closed_returns)} ({wins / len(closed_returns) * 100:.0f}%)"
            )

        with st.expander(f"📋 Historik ({len(closed_trades)} lukkede handler)"):
            closed_rows = []
            for t in sorted(
                closed_trades, key=lambda x: x.get("exit_date", ""), reverse=True
            ):
                ret = t.get("return_pct")
                arrow = ("🟢" if ret >= 0 else "🔴") if ret is not None else "—"
                closed_rows.append(
                    {
                        "Aktie": t["ticker"],
                        "Earnings dato": t.get("earnings_date", "?"),
                        "Surprise": f"+{t['surprise_pct']:.1f}%"
                        if t.get("surprise_pct")
                        else "?",
                        "Købt": t.get("entry_date", "?"),
                        "Solgt": t.get("exit_date", "?"),
                        "Dage holdt": t.get("hold_days", "?"),
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
# Manuel portfolio tracker
# ------------------------------------------------------------------
st.subheader("💼 Min PEAD-portefølje (manuel)")
st.caption("Hold styr på dine egne PEAD-trades baseret på signalerne ovenfor")

portfolio = _load_portfolio()
portfolio_set = {p["ticker"].upper() for p in portfolio}

# Tilføj position
with st.expander("➕ Tilføj position"):
    col1, col2, col3 = st.columns(3)
    new_ticker = col1.text_input("Ticker", key="pead_new_ticker").upper().strip()
    new_date = col2.date_input("Købt dato", value=date.today(), key="pead_new_date")
    new_price = col3.number_input(
        "Købt til ($)", min_value=0.0, step=0.01, key="pead_new_price"
    )

    col4, col5 = st.columns(2)
    new_surprise = col4.number_input(
        "EPS surprise (%)", min_value=0.0, step=0.1, key="pead_surprise"
    )
    new_earnings_date = col5.date_input(
        "Earnings dato", value=date.today(), key="pead_earnings_date"
    )

    if st.button("Tilføj", key="pead_add_btn"):
        if new_ticker and new_ticker not in portfolio_set:
            portfolio.append(
                {
                    "ticker": new_ticker,
                    "buy_date": str(new_date),
                    "buy_price": new_price if new_price > 0 else None,
                    "surprise_pct": new_surprise if new_surprise > 0 else None,
                    "earnings_date": str(new_earnings_date),
                }
            )
            _save_portfolio(portfolio)
            st.success(f"Tilføjet: {new_ticker}")
            st.rerun()
        elif new_ticker in portfolio_set:
            st.warning(f"{new_ticker} er allerede i porteføljen.")

# Vis portefølje
if portfolio:
    all_tickers = tuple(p["ticker"] for p in portfolio)
    current_prices = _fetch_prices(all_tickers)

    port_rows = []
    for p in sorted(portfolio, key=lambda x: x["ticker"]):
        ticker = p["ticker"]
        buy_price = p.get("buy_price")
        current = current_prices.get(ticker)

        if buy_price and current:
            ret_pct = (current / buy_price - 1) * 100
            arrow = "🟢" if ret_pct >= 0 else "🔴"
            ret_str = f"{arrow} {ret_pct:+.1f}%"
        else:
            ret_str = "—"

        # Dage siden køb
        try:
            days = (date.today() - date.fromisoformat(p["buy_date"])).days
        except Exception:
            days = "?"

        port_rows.append(
            {
                "Aktie": ticker,
                "Earnings dato": p.get("earnings_date", "?"),
                "Surprise": f"+{p['surprise_pct']:.1f}%"
                if p.get("surprise_pct")
                else "?",
                "Købt": p.get("buy_date", "?"),
                "Dage holdt": f"{days}/{hold_days}",
                "Købt @": f"${buy_price:.2f}" if buy_price else "—",
                "Kurs nu": f"${current:.2f}" if current else "—",
                "Afkast": ret_str,
            }
        )

    returns_port = []
    for p in portfolio:
        bp = p.get("buy_price")
        cp = current_prices.get(p["ticker"])
        if bp and cp:
            returns_port.append((cp / bp - 1) * 100)

    if returns_port:
        c1, c2, c3 = st.columns(3)
        c1.metric("Positioner", len(portfolio))
        c2.metric("Snit afkast", f"{sum(returns_port) / len(returns_port):+.1f}%")
        c3.metric(
            "Bedst / Dårligst", f"{max(returns_port):+.1f}% / {min(returns_port):+.1f}%"
        )

    st.dataframe(pd.DataFrame(port_rows), use_container_width=True, hide_index=True)

    # Slet position
    with st.expander("🗑️ Fjern position"):
        to_remove = st.selectbox(
            "Vælg aktie at fjerne",
            options=[p["ticker"] for p in portfolio],
            key="pead_remove_select",
        )
        if st.button("Fjern", key="pead_remove_btn"):
            portfolio = [p for p in portfolio if p["ticker"] != to_remove]
            _save_portfolio(portfolio)
            st.success(f"Fjernet: {to_remove}")
            st.rerun()
else:
    st.info("Din PEAD-portefølje er tom. Tilføj en position ovenfor.")

st.divider()

# ------------------------------------------------------------------
# Positionsstørrelse-beregner
# ------------------------------------------------------------------
st.subheader("🧮 Positionsstørrelse-beregner")
st.caption(
    f"PEAD-positioner holdes typisk {hold_days} dage — brug dette til at beregne din eksponering."
)

col_sel, col_amt = st.columns([2, 1])
with col_sel:
    all_candidates = [s["ticker"] for s in surprises] + [p["ticker"] for p in portfolio]
    calc_tickers = list(dict.fromkeys(all_candidates))  # unik rækkefølge
    selected = st.multiselect(
        "Vælg aktier",
        options=calc_tickers if calc_tickers else ["Ingen kandidater endnu"],
        key="pead_calc_select",
    )
with col_amt:
    currency = st.selectbox("Valuta", ["USD", "DKK"], index=1, key="pead_currency")
    total_amount = st.number_input(
        f"Samlet beløb ({currency})",
        min_value=0.0,
        value=50_000.0 if currency == "DKK" else 10_000.0,
        step=1000.0,
        format="%.0f",
        key="pead_amount",
    )

if selected and total_amount > 0:
    calc_prices = _fetch_prices(tuple(selected))
    equal_weight = total_amount / len(selected)

    calc_rows = []
    for ticker in selected:
        price = calc_prices.get(ticker)
        amount = equal_weight
        shares = amount / price if price else None
        calc_rows.append(
            {
                "Aktie": ticker,
                f"Beløb ({currency})": f"{amount:,.0f}",
                "Kurs ($)": f"${price:.2f}" if price else "—",
                "Antal aktier": f"{shares:.1f}" if shares else "—",
            }
        )

    st.dataframe(
        pd.DataFrame(calc_rows),
        use_container_width=True,
        hide_index=True,
    )
    c1, c2 = st.columns(2)
    c1.metric("Antal positioner", len(selected))
    c2.metric(f"Samlet beløb ({currency})", f"{total_amount:,.0f}")

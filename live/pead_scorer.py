"""live/pead_scorer.py

PEAD (Post-Earnings Announcement Drift) live scorer.

Kør manuelt eller via cron (anbefalet: dagligt kl. 8 og 18):
    uv run python live/pead_scorer.py

Hvad den gør:
  1. Scanner S&P 500 for aktier med positiv EPS surprise inden for de seneste 5 dage
  2. Åbner paper trade-positioner for nye kandidater
  3. Lukker positioner der har været holdt i HOLD_DAYS handelsdage
  4. Gemmer status til live/data/pead_ranking.json
  5. Gemmer paper trades til live/data/pead_paper_trades.json
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "research"))

from engine.data import get_sp500_tickers
from research.strategies.pead import (
    get_earnings_surprises,
    get_upcoming_earnings,
    HOLD_DAYS,
    TOP_N,
    SURPRISE_THRESHOLD,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Konfiguration
# ------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

RANKING_FILE = DATA_DIR / "pead_ranking.json"
PAPER_FILE = DATA_DIR / "pead_paper_trades.json"


# ------------------------------------------------------------------
# Paper trade helpers
# ------------------------------------------------------------------


def _load_paper_trades() -> dict:
    if not PAPER_FILE.exists():
        return {"positions": {}, "closed_trades": [], "equity_history": []}
    try:
        return json.loads(PAPER_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"positions": {}, "closed_trades": [], "equity_history": []}


def _save_paper_trades(pt: dict) -> None:
    PAPER_FILE.write_text(
        json.dumps(pt, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _get_prices(tickers: list[str]) -> dict[str, float]:
    """Hent seneste close for tickers via yfinance."""
    if not tickers:
        return {}
    try:
        raw = yf.download(tickers, period="2d", auto_adjust=True, progress=False)
        if raw.empty:
            return {}
        close = raw["Close"] if "Close" in raw.columns else raw
        if isinstance(close, pd.Series):
            close = close.to_frame(name=tickers[0])
        last = close.ffill().iloc[-1]
        return {
            str(t): float(last[t])
            for t in tickers
            if t in last.index and not pd.isna(last[t])
        }
    except Exception as e:
        logger.warning("Kunne ikke hente priser: %s", e)
        return {}


def _trading_days_held(entry_date_str: str) -> int:
    """Antal handelsdage siden entry_date (ekskl. weekender, simpel approx)."""
    try:
        entry = date.fromisoformat(entry_date_str)
        today = date.today()
        days = 0
        current = entry
        while current < today:
            current += timedelta(days=1)
            if current.weekday() < 5:  # mandag-fredag
                days += 1
        return days
    except Exception:
        return 0


def _update_paper_trades(candidates: pd.DataFrame) -> None:
    """
    Opdater PEAD paper trade journal.
    - Luk positioner der har nået HOLD_DAYS handelsdage
    - Åbn nye positioner for nylige EPS-surprises (op til TOP_N samtidige)
    """
    pt = _load_paper_trades()
    today_str = str(date.today())
    positions: dict = pt["positions"]

    # --- Luk positioner der har nået HOLD_DAYS ---
    to_close = [
        ticker
        for ticker, pos in positions.items()
        if _trading_days_held(pos["entry_date"]) >= HOLD_DAYS
    ]

    if to_close:
        prices = _get_prices(to_close)
        for ticker in to_close:
            pos = positions[ticker]
            exit_price = prices.get(ticker)
            entry_price = pos.get("entry_price")
            ret_pct = (
                round((exit_price / entry_price - 1) * 100, 2)
                if exit_price and entry_price
                else None
            )
            pt["closed_trades"].append(
                {
                    "ticker": ticker,
                    "entry_date": pos["entry_date"],
                    "entry_price": entry_price,
                    "exit_date": today_str,
                    "exit_price": exit_price,
                    "return_pct": ret_pct,
                    "hold_days": _trading_days_held(pos["entry_date"]),
                    "earnings_date": pos.get("earnings_date"),
                    "surprise_pct": pos.get("surprise_pct"),
                }
            )
            del positions[ticker]
            logger.info(
                "PEAD SÆLG %s @ %.2f (%.1f%%, %d dage)",
                ticker,
                exit_price or 0,
                ret_pct or 0,
                _trading_days_held(pos["entry_date"]),
            )

    # --- Åbn nye positioner ---
    open_count = len(positions)
    new_entries = []

    for _, row in candidates.iterrows():
        if open_count >= TOP_N:
            break
        ticker = row["ticker"]
        if ticker in positions:
            continue  # allerede åben
        new_entries.append(ticker)
        open_count += 1

    if new_entries:
        prices = _get_prices(new_entries)
        for _, row in candidates.iterrows():
            ticker = row["ticker"]
            if ticker not in new_entries:
                continue
            entry_price = prices.get(ticker)
            positions[ticker] = {
                "entry_date": today_str,
                "entry_price": entry_price,
                "earnings_date": str(row["earnings_date"]),
                "surprise_pct": float(row["surprise_pct"]),
                "company_name": row.get("company_name", ticker),
                "actual_eps": float(row["actual_eps"]),
                "estimated_eps": float(row["estimated_eps"]),
            }
            logger.info(
                "PEAD KØB %s @ %.2f  (surprise: +%.1f%%)",
                ticker,
                entry_price or 0,
                row["surprise_pct"],
            )

    # --- Equity snapshot ---
    all_tickers = list(positions.keys())
    prices_all = _get_prices(all_tickers) if all_tickers else {}
    returns = []
    for ticker, pos in positions.items():
        p = prices_all.get(ticker)
        ep = pos.get("entry_price")
        if p and ep:
            returns.append((p / ep - 1) * 100)

    avg_ret = round(sum(returns) / len(returns), 2) if returns else 0.0
    pt["equity_history"].append(
        {
            "date": today_str,
            "open_positions": len(positions),
            "avg_return_pct": avg_ret,
        }
    )

    pt["positions"] = positions
    _save_paper_trades(pt)
    logger.info(
        "PEAD paper trades opdateret: %d åbne, %d lukkede",
        len(positions),
        len(pt["closed_trades"]),
    )


# ------------------------------------------------------------------
# Hoved
# ------------------------------------------------------------------


def main() -> None:
    today = date.today()

    if today.weekday() >= 5:
        logger.info("Weekend — springer over")
        return

    tickers = get_sp500_tickers()
    logger.info("Scanner %d tickers for EPS surprises...", len(tickers))

    # Scan for earnings surprises
    surprises = get_earnings_surprises(tickers, lookback_days=5)
    logger.info(
        "Fandt %d aktier med positiv EPS surprise (>%.0f%%)",
        len(surprises),
        SURPRISE_THRESHOLD * 100,
    )

    # Kommende earnings (til info)
    upcoming = get_upcoming_earnings(tickers, days_ahead=7)

    # Gem ranking JSON
    output = {
        "updated_at": datetime.now().isoformat(),
        "as_of_date": str(today),
        "universe_size": len(tickers),
        "surprise_threshold_pct": round(SURPRISE_THRESHOLD * 100, 1),
        "hold_days": HOLD_DAYS,
        "surprises": surprises.to_dict(orient="records"),
        "upcoming_earnings": upcoming.head(20).to_dict(orient="records"),
    }

    # Serialiser dates
    for entry in output["surprises"]:
        if hasattr(entry.get("earnings_date"), "isoformat"):
            entry["earnings_date"] = entry["earnings_date"].isoformat()
    for entry in output["upcoming_earnings"]:
        if hasattr(entry.get("earnings_date"), "isoformat"):
            entry["earnings_date"] = entry["earnings_date"].isoformat()

    with open(RANKING_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info("PEAD ranking gemt: %s", RANKING_FILE)

    if not surprises.empty:
        logger.info(
            "Top-3 surprises: %s",
            ", ".join(
                f"{r['ticker']} (+{r['surprise_pct']:.1f}%)"
                for _, r in surprises.head(3).iterrows()
            ),
        )

    # Opdater paper trades
    _update_paper_trades(surprises)


if __name__ == "__main__":
    main()

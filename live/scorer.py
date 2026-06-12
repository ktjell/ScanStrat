"""live/scorer.py

Kores 4x om dagen (fx kl. 8, 12, 18, 22) via macOS launchd eller cron.

Hvad den goer:
  1. Henter seneste kursdata fra yfinance (S&P 500)
  2. Loader model fra disk hvis den er traenet for nylig — ellers retraener
  3. Beregner ML-score (P(slaar SPY naeste uge)) for alle aktier
  4. Gemmer top-30 + metadata til live/data/latest_ranking.json
  5. Gemmer/opdaterer model til live/data/model.pkl (kun ved retraening)

Kores med:
    uv run python live/scorer.py
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "research"))

from engine.data import get_sp500_tickers, CacheManager, DataService, YFinanceLoader
from engine.config import Settings
from research.strategies.ml_ranker import (
    MLRankerStrategy,
    XGB_PARAMS,
    MIN_TRAIN_WEEKS,
    RETRAIN_WEEKS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Konfiguration
# ------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

RANKING_FILE = DATA_DIR / "latest_ranking.json"
MODEL_FILE = DATA_DIR / "model.pkl"
PAPER_FILE = DATA_DIR / "paper_trades.json"
TOP_N_OUTPUT = 30  # gem top-30 i JSON (dashboard viser top-15)
RETRAIN_DAYS = 28  # retrain model hver 28 dage


# ------------------------------------------------------------------
# Paper trading helpers
# ------------------------------------------------------------------


def _load_paper_trades() -> dict:
    """Hent paper trade tilstand fra disk."""
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


def _get_current_prices(tickers: list[str]) -> dict[str, float]:
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
        prices = {}
        for t in tickers:
            if t in close.columns:
                series = close[t].dropna()
                if not series.empty:
                    prices[t] = float(series.iloc[-1])
        return prices
    except Exception as e:
        logger.warning("Kunne ikke hente paper trade priser: %s", e)
        return {}


def _update_paper_trades(new_top15: list[str]) -> None:
    """
    Opdater paper trade journal baseret på ny top-15 liste.
    - Nye tickers: åbn position til dagskurs
    - Udgåede tickers: luk position til dagskurs
    - Log equity snapshot
    """
    pt = _load_paper_trades()
    today_str = str(date.today())
    positions = pt["positions"]

    exiting = [t for t in positions if t not in new_top15]
    entering = [t for t in new_top15 if t not in positions]

    all_needed = list(set(new_top15) | set(exiting))
    prices = _get_current_prices(all_needed)

    # Luk udgående positioner
    for ticker in exiting:
        entry = positions[ticker]
        exit_price = prices.get(ticker)
        if exit_price and entry.get("entry_price"):
            ret_pct = round((exit_price / entry["entry_price"] - 1) * 100, 2)
        else:
            ret_pct = None
        pt["closed_trades"].append(
            {
                "ticker": ticker,
                "entry_date": entry["entry_date"],
                "entry_price": entry.get("entry_price"),
                "exit_date": today_str,
                "exit_price": exit_price,
                "return_pct": ret_pct,
            }
        )
        del positions[ticker]
        logger.info(
            "Paper SÆLG %s @ %.2f (%.1f%%)", ticker, exit_price or 0, ret_pct or 0
        )

    # Åbn nye positioner
    for ticker in entering:
        entry_price = prices.get(ticker)
        positions[ticker] = {"entry_date": today_str, "entry_price": entry_price}
        logger.info("Paper KØB %s @ %.2f", ticker, entry_price or 0)

    # Equity snapshot (gennemsnit af åbne positioners afkast)
    returns = []
    for ticker, pos in positions.items():
        p = prices.get(ticker)
        if p and pos.get("entry_price"):
            returns.append((p / pos["entry_price"] - 1) * 100)
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
        "Paper trades opdateret: %d åbne, %d lukkede",
        len(positions),
        len(pt["closed_trades"]),
    )


# ------------------------------------------------------------------
# Model helpers
# ------------------------------------------------------------------


def _load_model() -> MLRankerStrategy | None:
    """Load gemt model fra disk. Returnerer None hvis fil ikke eksisterer eller er for gammel."""
    if not MODEL_FILE.exists():
        return None
    try:
        with open(MODEL_FILE, "rb") as f:
            obj = pickle.load(f)
        logger.info("Model loadet fra disk (traenet %s)", obj.get("trained_at"))
        strat: MLRankerStrategy = obj["strategy"]
        return strat
    except Exception as e:
        logger.warning("Kunne ikke loade model: %s", e)
        return None


def _save_model(strat: MLRankerStrategy) -> None:
    with open(MODEL_FILE, "wb") as f:
        pickle.dump({"strategy": strat, "trained_at": datetime.now().isoformat()}, f)
    logger.info("Model gemt til %s", MODEL_FILE)


def _model_needs_retrain(strat: MLRankerStrategy | None) -> bool:
    """Retrain hvis model ikke eksisterer eller er mere end RETRAIN_DAYS gammel."""
    if strat is None or strat._model is None:
        return True
    if strat._last_trained is None:
        return True
    age = pd.Timestamp.now() - strat._last_trained
    return age.days >= RETRAIN_DAYS


def _fetch_data(
    tickers: list[str], lookback_days: int = 400
) -> dict[str, pd.DataFrame]:
    """Hent kursdata for alle tickers via DataService (disk-cache + yfinance)."""
    settings = Settings.default()
    service = DataService(YFinanceLoader(), CacheManager(settings.cache), settings)
    start = date.today() - timedelta(days=lookback_days)
    end = date.today()
    logger.info("Henter kursdata for %d tickers...", len(tickers))
    data = service.get_batch(tickers, start, end)
    logger.info("  -> %d tickers med data", len(data))
    return data


# ------------------------------------------------------------------
# Hoved
# ------------------------------------------------------------------


def main() -> None:
    today = date.today()

    # Spring over i weekender (markedet er lukket)
    if today.weekday() >= 5:
        logger.info("Weekend — springer over")
        return

    tickers = get_sp500_tickers()

    # Hent data
    data = _fetch_data(tickers, lookback_days=400)

    # Sikkerhedstjek: hvis vi har for lidt data er noget gået galt (rate-limit, netværk osv.)
    # Bevar den eksisterende ranking i stedet for at overskrive med dårlig data
    MIN_TICKERS = max(200, len(tickers) // 2)  # mindst halvdelen af universet
    if len(data) < MIN_TICKERS:
        logger.warning(
            "For lidt data: %d tickers (minimum %d). "
            "Springer over — beholder eksisterende ranking.",
            len(data),
            MIN_TICKERS,
        )
        return

    # Tilfoej SPY til data-dict hvis ikke allerede der
    if "SPY" not in data:
        logger.info("Henter SPY...")
        start = date.today() - timedelta(days=400)
        spy_raw = yf.download(
            "SPY", start=str(start), end=str(today), auto_adjust=True, progress=False
        )
        if not spy_raw.empty:
            df = spy_raw[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.columns = ["open", "high", "low", "close", "volume"]
            df.index = pd.to_datetime(df.index)
            data["SPY"] = df

    # Load eller opret strategi
    strat = _load_model()
    needs_retrain = _model_needs_retrain(strat)

    if needs_retrain:
        logger.info("Traener model (kan tage 1-2 min)...")
        strat = MLRankerStrategy(top_n=15)
        # Kald rank() med al tilgaengelig historik for at traene
        _ = strat.rank(data, as_of=today)
        if strat._model is not None:
            _save_model(strat)
            logger.info("Model traenet og gemt")
        else:
            logger.error("Model traening fejlede — for lidt data?")
            return
    else:
        # Brug gemt model men opdater feature store med nye data
        _ = strat.rank(data, as_of=today)

    # Hent ranking
    ranked = strat.rank(data, as_of=today)

    if ranked.empty:
        logger.error("Ingen ranking output — tjek data")
        return

    # Gem forrige top-15 inden vi overskriver (bruges af dashboard til at vise NY/SOLGT)
    previous_top15: list[str] = []
    if RANKING_FILE.exists():
        try:
            prev = json.loads(RANKING_FILE.read_text(encoding="utf-8"))
            previous_top15 = [s["ticker"] for s in prev.get("top_stocks", [])[:15]]
        except Exception:
            pass

    # Byg output
    top = ranked.head(TOP_N_OUTPUT).copy()
    top["ml_score_pct"] = (top["ml_score"] * 100).round(1)

    output = {
        "updated_at": datetime.now().isoformat(),
        "as_of_date": str(today),
        "model_trained_at": strat._last_trained.isoformat()
        if strat._last_trained
        else None,
        "universe_size": len(data),
        "previous_top15": previous_top15,
        "top_stocks": [
            {
                "rank": int(row["rank"]),
                "ticker": row["ticker"],
                "ml_score": round(float(row["ml_score"]), 4),
                "ml_score_pct": float(row["ml_score_pct"]),
            }
            for _, row in top.iterrows()
        ],
    }

    with open(RANKING_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(
        "Ranking gemt: %s  |  Top-3: %s",
        RANKING_FILE,
        ", ".join(
            f"{r['ticker']} ({r['ml_score_pct']}%)" for r in output["top_stocks"][:3]
        ),
    )

    # Opdater paper trades (køb/sælg baseret på ny top-15)
    new_top15 = [r["ticker"] for r in output["top_stocks"][:15]]
    _update_paper_trades(new_top15)


if __name__ == "__main__":
    main()

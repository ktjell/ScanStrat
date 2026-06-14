"""live/ensemble_scorer.py

Ensemble Ranker live scorer — kører parallelt med scorer.py.

Hvad den gør:
  1. Henter kursdata (deler cache med scorer.py)
  2. Loader/træner ensemble-modellen (3 specialiserede XGBoost-modeller)
  3. Gemmer top-30 + individuelle model-scores til live/data/ensemble_ranking.json
  4. Opdaterer paper trades i live/data/ensemble_paper_trades.json

Kør med:
    uv run python live/ensemble_scorer.py
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
from research.strategies.ensemble_ranker import EnsembleRankerStrategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Konfiguration
# ------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

RANKING_FILE = DATA_DIR / "ensemble_ranking.json"
MODEL_FILE = DATA_DIR / "ensemble_model.pkl"
PAPER_FILE = DATA_DIR / "ensemble_paper_trades.json"
TOP_N_OUTPUT = 30
RETRAIN_DAYS = 28


# ------------------------------------------------------------------
# Model helpers
# ------------------------------------------------------------------


def _load_model() -> EnsembleRankerStrategy | None:
    if not MODEL_FILE.exists():
        return None
    try:
        with open(MODEL_FILE, "rb") as f:
            obj = pickle.load(f)
        strat: EnsembleRankerStrategy = obj["strategy"]
        logger.info("Ensemble model loadet fra disk (trænet %s)", obj.get("trained_at"))
        return strat
    except Exception as e:
        logger.warning("Kunne ikke loade ensemble model: %s", e)
        return None


def _save_model(strat: EnsembleRankerStrategy) -> None:
    with open(MODEL_FILE, "wb") as f:
        pickle.dump({"strategy": strat, "trained_at": datetime.now().isoformat()}, f)
    logger.info("Ensemble model gemt til %s", MODEL_FILE)


def _model_needs_retrain(strat: EnsembleRankerStrategy | None) -> bool:
    if strat is None or any(m is None for m in strat._models.values()):
        return True
    if strat._last_trained is None:
        return True
    age = pd.Timestamp.now() - strat._last_trained
    return age.days >= RETRAIN_DAYS


def _fetch_data(
    tickers: list[str], lookback_days: int = 400
) -> dict[str, pd.DataFrame]:
    settings = Settings.default()
    service = DataService(YFinanceLoader(), CacheManager(settings.cache), settings)
    start = date.today() - timedelta(days=lookback_days)
    end = date.today()
    logger.info("Henter kursdata for %d tickers...", len(tickers))
    data = service.get_batch(tickers, start, end)
    logger.info("  -> %d tickers med data", len(data))
    return data


# ------------------------------------------------------------------
# Paper trading helpers (identisk med scorer.py)
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


def _get_current_prices(tickers: list[str]) -> dict[str, float]:
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


def _update_paper_trades(new_top15: list[str]) -> None:
    pt = _load_paper_trades()
    today_str = str(date.today())
    positions = pt["positions"]

    exiting = [t for t in positions if t not in new_top15]
    entering = [t for t in new_top15 if t not in positions]
    all_needed = list(set(new_top15) | set(exiting))
    prices = _get_current_prices(all_needed)

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
            "Ensemble SÆLG %s @ %.2f (%.1f%%)", ticker, exit_price or 0, ret_pct or 0
        )

    for ticker in entering:
        entry_price = prices.get(ticker)
        positions[ticker] = {"entry_date": today_str, "entry_price": entry_price}
        logger.info("Ensemble KØB %s @ %.2f", ticker, entry_price or 0)

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
        "Ensemble paper trades opdateret: %d åbne, %d lukkede",
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
    data = _fetch_data(tickers, lookback_days=400)

    MIN_TICKERS = max(200, len(tickers) // 2)
    if len(data) < MIN_TICKERS:
        logger.warning(
            "For lidt data: %d tickers (minimum %d). Springer over.",
            len(data),
            MIN_TICKERS,
        )
        return

    # Tilføj SPY
    if "SPY" not in data:
        logger.info("Henter SPY...")
        start = today - timedelta(days=400)
        spy_raw = yf.download(
            "SPY", start=str(start), end=str(today), auto_adjust=True, progress=False
        )
        if not spy_raw.empty:
            df = spy_raw[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.columns = ["open", "high", "low", "close", "volume"]
            df.index = pd.to_datetime(df.index)
            data["SPY"] = df

    strat = _load_model()
    needs_retrain = _model_needs_retrain(strat)

    if needs_retrain:
        logger.info("Træner ensemble model (kan tage 2-3 min)...")
        strat = EnsembleRankerStrategy(top_n=15)
        _ = strat.rank(data, as_of=today)
        if strat._last_trained is not None:
            _save_model(strat)
            logger.info("Ensemble model trænet og gemt")
        else:
            logger.error("Ensemble træning fejlede")
            return
    else:
        _ = strat.rank(data, as_of=today)

    ranked = strat.rank(data, as_of=today)

    if ranked.empty:
        logger.error("Ingen ensemble ranking output")
        return

    # Gem forrige top-15
    previous_top15: list[str] = []
    if RANKING_FILE.exists():
        try:
            prev = json.loads(RANKING_FILE.read_text(encoding="utf-8"))
            previous_top15 = [s["ticker"] for s in prev.get("top_stocks", [])[:15]]
        except Exception:
            pass

    top = ranked.head(TOP_N_OUTPUT).copy()
    top["ensemble_score_pct"] = (top["ensemble_score"] * 100).round(1)

    output = {
        "updated_at": datetime.now().isoformat(),
        "as_of_date": str(today),
        "model_trained_at": strat._last_trained.isoformat()
        if strat._last_trained
        else None,
        "universe_size": len(data),
        "previous_top15": previous_top15,
        "model_weights": strat._weights,
        "top_stocks": [
            {
                "rank": int(row["rank"]),
                "ticker": row["ticker"],
                "ensemble_score": round(float(row["ensemble_score"]), 4),
                "ensemble_score_pct": float(row["ensemble_score_pct"]),
                "momentum_score": round(float(row["momentum_score"]), 4)
                if not pd.isna(row["momentum_score"])
                else None,
                "technical_score": round(float(row["technical_score"]), 4)
                if not pd.isna(row["technical_score"])
                else None,
                "macro_score": round(float(row["macro_score"]), 4)
                if not pd.isna(row["macro_score"])
                else None,
            }
            for _, row in top.iterrows()
        ],
    }

    with open(RANKING_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(
        "Ensemble ranking gemt  |  Top-3: %s",
        ", ".join(
            f"{r['ticker']} ({r['ensemble_score_pct']}%)"
            for r in output["top_stocks"][:3]
        ),
    )

    new_top15 = [r["ticker"] for r in output["top_stocks"][:15]]
    _update_paper_trades(new_top15)


if __name__ == "__main__":
    main()

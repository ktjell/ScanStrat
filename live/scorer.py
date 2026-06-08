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
TOP_N_OUTPUT = 30  # gem top-30 i JSON (dashboard viser top-15)
RETRAIN_DAYS = 28  # retrain model hver 28 dage


# ------------------------------------------------------------------
# Hjælpefunktioner
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


if __name__ == "__main__":
    main()

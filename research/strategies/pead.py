"""research/strategies/pead.py

Post-Earnings Announcement Drift (PEAD) strategi.

Idé: Aktier der slår earnings-estimater med >SURPRISE_THRESHOLD fortsætter
typisk med at stige de næste 10-30 dage (markedet under-reagerer på gode nyheder).

Signal-logik:
  1. Find aktier i universet der har rapporteret earnings inden for de seneste LOOKBACK_DAYS
  2. Beregn EPS surprise = (faktisk EPS - estimeret EPS) / |estimeret EPS|
  3. Tag de aktier med størst positiv surprise (op til TOP_N)
  4. Hold position i HOLD_DAYS dage, derefter luk

Dette er et event-drevet overlay, ikke en klassisk rank-baseret strategi.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Parametre
# ------------------------------------------------------------------
TOP_N: int = 10  # max antal samtidige positioner
SURPRISE_THRESHOLD: float = 0.03  # mindst 3% positiv EPS surprise
HOLD_DAYS: int = 20  # hold position i 20 handelsdage
LOOKBACK_DAYS: int = 5  # kig X kalenderdage tilbage for earnings


def get_earnings_surprises(
    tickers: list[str],
    lookback_days: int = LOOKBACK_DAYS,
) -> pd.DataFrame:
    """
    Scan tickers for positive EPS surprises inden for de seneste lookback_days.

    Returnerer DataFrame med kolonner:
        ticker, earnings_date, actual_eps, estimated_eps, surprise_pct, company_name
    Sorteret efter surprise_pct descending.
    """
    cutoff = date.today() - timedelta(days=lookback_days)
    rows = []

    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            ed = t.earnings_dates
            if ed is None or ed.empty:
                continue

            # Normaliser index til tz-naive dates
            ed = ed.copy()
            if ed.index.tz is not None:
                ed.index = ed.index.tz_localize(None)
            ed.index = pd.to_datetime(ed.index)

            # Kig kun på earnings inden for vinduet
            recent = ed[(ed.index.date >= cutoff) & (ed.index.date <= date.today())]
            if recent.empty:
                continue

            for dt, row in recent.iterrows():
                actual = row.get("Reported EPS")
                estimated = row.get("EPS Estimate")

                if pd.isna(actual) or pd.isna(estimated) or estimated == 0:
                    continue

                surprise_pct = float((actual - estimated) / abs(estimated))

                if surprise_pct < SURPRISE_THRESHOLD:
                    continue

                info = t.info
                rows.append(
                    {
                        "ticker": ticker,
                        "earnings_date": dt.date(),
                        "actual_eps": round(float(actual), 4),
                        "estimated_eps": round(float(estimated), 4),
                        "surprise_pct": round(surprise_pct * 100, 2),
                        "company_name": info.get("shortName", ticker),
                    }
                )
        except Exception as e:
            logger.debug("Springer %s over: %s", ticker, e)
            continue

    if not rows:
        return pd.DataFrame(
            columns=[
                "ticker",
                "earnings_date",
                "actual_eps",
                "estimated_eps",
                "surprise_pct",
                "company_name",
            ]
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("surprise_pct", ascending=False).drop_duplicates("ticker")
    return df.reset_index(drop=True)


def get_upcoming_earnings(tickers: list[str], days_ahead: int = 7) -> pd.DataFrame:
    """
    Find aktier med planlagte earnings inden for de næste days_ahead dage.
    Nyttigt til at vide hvad der snart rapporterer.
    """
    cutoff_from = date.today()
    cutoff_to = date.today() + timedelta(days=days_ahead)
    rows = []

    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            ed = t.earnings_dates
            if ed is None or ed.empty:
                continue

            ed = ed.copy()
            if ed.index.tz is not None:
                ed.index = ed.index.tz_localize(None)
            ed.index = pd.to_datetime(ed.index)

            upcoming = ed[(ed.index.date >= cutoff_from) & (ed.index.date <= cutoff_to)]
            if upcoming.empty:
                continue

            for dt, row in upcoming.iterrows():
                info = t.info
                rows.append(
                    {
                        "ticker": ticker,
                        "earnings_date": dt.date(),
                        "company_name": info.get("shortName", ticker),
                        "estimated_eps": row.get("EPS Estimate"),
                    }
                )
        except Exception:
            continue

    if not rows:
        return pd.DataFrame(
            columns=["ticker", "earnings_date", "company_name", "estimated_eps"]
        )

    df = pd.DataFrame(rows)
    return df.sort_values("earnings_date").reset_index(drop=True)

"""live/macro_indicators.py

Henter og evaluerer 4 makro-indikatorer der advarer om markedskrak/recession.
Alle data hentes gratis fra FRED (Federal Reserve) og yfinance — ingen API-nøgle.

Indikatorer:
  1. Yield curve (T10Y2Y): 10-årig minus 2-årig rente
  2. High Yield credit spread (BAMLH0A0HYM2): junk bonds vs. investment grade
  3. Leading Economic Index (USSLIND): Conference Board's sammensatte indikator
  4. SPY vs. SMA200: bull/bear regime

Brug:
    from live.macro_indicators import get_all_indicators
    indicators = get_all_indicators()
"""

from __future__ import annotations

from datetime import date, timedelta
from io import StringIO

import logging

import numpy as np
import pandas as pd
import requests
import yfinance as yf

logging.basicConfig(level=logging.WARNING, stream=__import__("sys").stdout)
logger = logging.getLogger(__name__)

# FRED offentlige CSV-URL (ingen API-nøgle nødvendig)
_FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
# yfinance-ticker alternativer til FRED (bruges som fallback)
_FRED_YF_FALLBACK = {
    "T10Y2Y": "^TNX",  # 10-årig rente (approximation)
    "BAMLH0A0HYM2": None,  # ingen yfinance-ækvivalent
    "USSLIND": None,
}
_TIMEOUT = 20


def _fetch_fred(series_id: str, lookback_days: int = 365 * 3) -> pd.Series:
    """Hent en FRED-tidsserie som pd.Series. Returnerer tom serie ved fejl."""
    url = _FRED_URL.format(series=series_id)
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; ScanStrat/1.0)"}
        r = requests.get(url, timeout=_TIMEOUT, headers=headers)
        r.raise_for_status()
        # FRED returnerer HTML ved fejl — tjek første linje
        first_line = r.text.strip().split("\n")[0]
        if "DATE" not in first_line:
            logger.debug(
                "FRED %s returnerede HTML/fejl-svar (bruger fallback)", series_id
            )
            return pd.Series(dtype=float)
        df = pd.read_csv(StringIO(r.text), parse_dates=["DATE"], index_col="DATE")
        s = df.iloc[:, 0].replace(".", np.nan).astype(float).dropna()
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=lookback_days)
        return s[s.index >= cutoff]
    except Exception as e:
        logger.debug("FRED %s fejl (bruger fallback): %s", series_id, e)
        return pd.Series(dtype=float)


def _fetch_yf(ticker: str, lookback_days: int = 400) -> pd.Series:
    """Hent close-kurs fra yfinance."""
    start = date.today() - timedelta(days=lookback_days)
    raw = yf.download(ticker, start=str(start), auto_adjust=True, progress=False)
    if raw.empty:
        return pd.Series(dtype=float)
    s = raw["Close"].squeeze().dropna()
    s.index = pd.to_datetime(s.index)
    return s


def _fetch_spy(lookback_days: int = 400) -> pd.Series:
    start = date.today() - timedelta(days=lookback_days)
    raw = yf.download("SPY", start=str(start), auto_adjust=True, progress=False)
    if raw.empty:
        return pd.Series(dtype=float)
    close = raw["Close"].squeeze().dropna()
    close.index = pd.to_datetime(close.index)
    return close


# ------------------------------------------------------------------
# De 4 indikatorer
# ------------------------------------------------------------------


def yield_curve() -> dict:
    """
    10-årig minus 13-ugers US Treasury rente (yfinance: ^TNX og ^IRX).
    Tilnærmet svarende til T10Y2Y fra FRED.
    Rød: inverteret (< 0) — historisk recession-signal inden for 6-18 mdr.
    Gul: 0 – 0.5% — flad kurve, advarsel
    Grøn: > 0.5% — normal, positiv hældning
    """
    # ^TNX = 10-year yield, ^IRX = 13-week T-bill (begge i % * 10 i yfinance)
    t10 = _fetch_yf("^TNX", lookback_days=400)
    t3m = _fetch_yf("^IRX", lookback_days=400)

    # yfinance returnerer yield som float med korrekt %, fx 4.25 = 4.25%
    if t10.empty or t3m.empty:
        # Prøv FRED som fallback
        s = _fetch_fred("T10Y2Y")
        if s.empty:
            return _error("Yield Curve (10Y-3M)")
        latest = float(s.iloc[-1])
        prev_month = float(s.iloc[-22]) if len(s) >= 22 else float("nan")
        trend = latest - prev_month if not np.isnan(prev_month) else 0.0
        spread_s = s
    else:
        # Align on common dates og beregn spread
        df = pd.concat({"t10": t10, "t3m": t3m}, axis=1).dropna()
        spread_s = df["t10"] - df["t3m"]
        latest = float(spread_s.iloc[-1])
        prev_month = float(spread_s.iloc[-22]) if len(spread_s) >= 22 else float("nan")
        trend = latest - prev_month if not np.isnan(prev_month) else 0.0

    if latest < 0:
        status = "ROED"
        signal = f"Inverteret ({latest:+.2f}%) — recession-advarsel"
    elif latest < 0.5:
        status = "GOEL"
        signal = f"Flad ({latest:+.2f}%) — neutral"
    else:
        status = "GROEN"
        signal = f"Normal ({latest:+.2f}%) — sund"

    return {
        "name": "Yield Curve (10Y-3M)",
        "value": round(latest, 3),
        "unit": "%",
        "status": status,
        "signal": signal,
        "trend": round(trend, 3),
        "series": {str(k.date()): float(v) for k, v in spread_s.tail(252).items()},
        "description": "Inverteret rentekurve har forudset alle US-recessioner siden 1970.",
    }


def credit_spread() -> dict:
    """
    High Yield credit spread proxy: HYG (junk bonds) vs. IEF (7-10Y treasury).
    Beregnes som negativ relativ performance — stiger = spreads udvider sig.
    Rød: spread > 4 pct.point bredere end gennemsnit (krak-niveau)
    Gul: 2-4 pct.point
    Grøn: under 2 pct.point

    Forsøger først FRED BAMLH0A0HYM2 for præcis spread.
    """
    # Prøv FRED først
    s = _fetch_fred("BAMLH0A0HYM2")
    if not s.empty:
        latest = float(s.iloc[-1])
        prev_month = float(s.iloc[-22]) if len(s) >= 22 else float("nan")
        trend = latest - prev_month if not np.isnan(prev_month) else 0.0
        if latest > 500:
            status, signal = "ROED", f"{latest:.0f} bp — markedsstress/krak"
        elif latest > 350:
            status, signal = "GOEL", f"{latest:.0f} bp — forhøjet risiko"
        else:
            status, signal = "GROEN", f"{latest:.0f} bp — normalt"
        return {
            "name": "HY Credit Spread",
            "value": round(latest, 1),
            "unit": "bp",
            "status": status,
            "signal": signal,
            "trend": round(trend, 1),
            "series": {str(k.date()): float(v) for k, v in s.tail(252).items()},
            "description": "Stiger kraftigt før og under recessions og børskrak.",
        }

    # Fallback: HYG vs. IEF relativ performance (proxy for spread-bevægelse)
    hyg = _fetch_yf("HYG", lookback_days=400)
    ief = _fetch_yf("IEF", lookback_days=400)
    if hyg.empty or ief.empty:
        return _error("HY Credit Spread")

    df = pd.concat({"hyg": hyg, "ief": ief}, axis=1).dropna()
    # Beregn 90-dages rolling spread: IEF outperformance over HYG (normaliseret)
    hyg_ret = df["hyg"].pct_change(90)
    ief_ret = df["ief"].pct_change(90)
    spread_proxy = (ief_ret - hyg_ret) * 100  # pct.point, positiv = HYG underperformer

    latest = float(spread_proxy.iloc[-1])
    hist_mean = float(spread_proxy.mean())
    relative = latest - hist_mean  # afvigelse fra historisk gennemsnit

    if relative > 4:
        status = "ROED"
        signal = f"HYG underperformer markant (+{relative:.1f}pp) — kreditstress"
    elif relative > 2:
        status = "GOEL"
        signal = f"HYG svækkes ({relative:+.1f}pp vs. gennemsnit) — advarsel"
    else:
        status = "GROEN"
        signal = f"HYG normal ({relative:+.1f}pp vs. gennemsnit)"

    return {
        "name": "HY Credit Spread (proxy)",
        "value": round(latest, 2),
        "unit": "pp (HYG vs IEF)",
        "status": status,
        "signal": signal,
        "trend": round(relative, 2),
        "series": {str(k.date()): float(v) for k, v in spread_proxy.tail(252).items()},
        "description": "IEF > HYG outperformance indikerer øget kreditrisiko-aversion.",
    }


def leading_economic_index() -> dict:
    """
    Leading economic indicator proxy baseret på XLI (industrials) og IYT (transport).
    Forsøger først FRED USSLIND (Conference Board LEI).
    Fallback: 3-måneders relativ performance af XLI vs. SPY (industrisektor som ledende indikator).
    Rød: XLI/SPY faldende kraftigt (>5% underperformance) i 3 måneder
    Gul: mild svækkelse (2-5%)
    Grøn: industrisektor holder trit eller outperformer
    """
    # Prøv FRED først
    s = _fetch_fred("USSLIND", lookback_days=365 * 5)
    if not s.empty and len(s) >= 7:
        latest = float(s.iloc[-1])
        six_months_ago = float(s.iloc[-7])
        rate_6m = (latest / six_months_ago - 1) * 2 * 100
        mom = s.pct_change().dropna()
        neg_streak = sum(
            1
            for _ in __import__("itertools").takewhile(
                lambda v: v < 0, reversed(mom.values.tolist())
            )
        )
        if rate_6m < -4 and neg_streak >= 6:
            status, signal = (
                "ROED",
                f"Falder ({rate_6m:.1f}% ann.) — {neg_streak} neg. mdr. i træk",
            )
        elif rate_6m < 0 or neg_streak >= 3:
            status, signal = "GOEL", f"Svækkes ({rate_6m:.1f}% ann.) — advarsel"
        else:
            status, signal = "GROEN", f"Stiger ({rate_6m:.1f}% ann.)"
        return {
            "name": "Leading Economic Index",
            "value": round(latest, 2),
            "unit": "indeks",
            "status": status,
            "signal": signal,
            "trend": round(rate_6m, 2),
            "series": {str(k.date()): float(v) for k, v in s.tail(60).items()},
            "description": "Conference Board's 10-komponent forudsigende indikator for US-økonomi.",
        }

    # Fallback: XLI (industrials ETF) vs. SPY — ledende sektor-indikator
    xli = _fetch_yf("XLI", lookback_days=400)
    spy = _fetch_yf("SPY", lookback_days=400)
    if xli.empty or spy.empty or len(xli) < 63:
        return _error("Industrisektor-indikator (LEI proxy)")

    df = pd.concat({"xli": xli, "spy": spy}, axis=1).dropna()
    # 3-måneders (63 handelsdage) relativ performance
    xli_ret = (df["xli"].iloc[-1] / df["xli"].iloc[-63] - 1) * 100
    spy_ret = (df["spy"].iloc[-1] / df["spy"].iloc[-63] - 1) * 100
    rel = xli_ret - spy_ret  # positiv = XLI outperformer

    # 1-måneds trend
    xli_1m = (df["xli"].iloc[-1] / df["xli"].iloc[-22] - 1) * 100
    spy_1m = (df["spy"].iloc[-1] / df["spy"].iloc[-22] - 1) * 100
    rel_1m = xli_1m - spy_1m

    if rel < -5 and rel_1m < -2:
        status = "ROED"
        signal = (
            f"Industri svækkes kraftigt ({rel:.1f}pp vs. SPY, 3M) — recession-advarsel"
        )
    elif rel < -2 or rel_1m < -1:
        status = "GOEL"
        signal = f"Industri underperformer ({rel:.1f}pp vs. SPY, 3M) — advarsel"
    else:
        status = "GROEN"
        signal = f"Industri normal ({rel:+.1f}pp vs. SPY, 3M)"

    series_rel = (df["xli"].pct_change(1) - df["spy"].pct_change(1)).cumsum() * 100
    return {
        "name": "Industrisektor (LEI proxy)",
        "value": round(rel, 2),
        "unit": "pp vs. SPY",
        "status": status,
        "signal": signal,
        "trend": round(rel_1m, 2),
        "series": {str(k.date()): float(v) for k, v in series_rel.tail(252).items()},
        "description": "XLI (industrials) relativ performance vs. SPY — ledende sektor-indikator.",
    }


def spy_regime() -> dict:
    """
    SPY vs. 200-dages SMA.
    Rød: SPY < SMA200 (bear market)
    Gul: SPY < SMA50 men > SMA200
    Grøn: SPY > SMA50 > SMA200 (bull market)
    """
    close = _fetch_spy()
    if close.empty or len(close) < 200:
        return _error("SPY vs. SMA200")

    latest = float(close.iloc[-1])
    sma50 = float(close.tail(50).mean())
    sma200 = float(close.tail(200).mean())

    pct_above_200 = (latest / sma200 - 1) * 100

    if latest < sma200:
        status = "ROED"
        signal = f"Under SMA200 ({pct_above_200:.1f}%) — bear market"
    elif latest < sma50:
        status = "GOEL"
        signal = f"Over SMA200 men under SMA50 — svækkelse"
    else:
        status = "GROEN"
        signal = f"Over SMA50 og SMA200 (+{pct_above_200:.1f}%) — bull market"

    return {
        "name": "SPY vs. SMA200",
        "value": round(latest, 2),
        "unit": "USD",
        "sma50": round(sma50, 2),
        "sma200": round(sma200, 2),
        "status": status,
        "signal": signal,
        "trend": round(pct_above_200, 2),
        "series": {str(k.date()): float(v) for k, v in close.tail(252).items()},
        "description": "SPY under 200-dages glidende gennemsnit = bekræftet bear market.",
    }


def _error(name: str) -> dict:
    return {
        "name": name,
        "value": None,
        "unit": "",
        "status": "GRAA",
        "signal": "Data ikke tilgængeligt",
        "trend": 0.0,
        "series": {},
        "description": "",
    }


# ------------------------------------------------------------------
# Samlet output
# ------------------------------------------------------------------


def get_all_indicators() -> list[dict]:
    """Hent alle 4 indikatorer. Returnerer liste med én dict per indikator."""
    return [
        yield_curve(),
        credit_spread(),
        leading_economic_index(),
        spy_regime(),
    ]


def overall_signal(indicators: list[dict]) -> tuple[str, str]:
    """
    Beregn samlet markedssignal baseret på alle 4 indikatorer.
    Returnerer (status, besked).
    """
    counts = {"ROED": 0, "GOEL": 0, "GROEN": 0, "GRAA": 0}
    for ind in indicators:
        counts[ind["status"]] += 1

    if counts["ROED"] >= 3:
        return "ROED", f"{counts['ROED']}/4 røde — gå defensiv (cash/guld)"
    elif counts["ROED"] >= 2:
        return (
            "GOEL",
            f"{counts['ROED']}/4 røde, {counts['GOEL']}/4 gule — reducer risiko",
        )
    elif counts["ROED"] >= 1 or counts["GOEL"] >= 2:
        return "GOEL", f"Blandet signal — vær forsigtig"
    else:
        return "GROEN", f"Alle indikatorer grønne — bull marked"


if __name__ == "__main__":
    inds = get_all_indicators()
    status, msg = overall_signal(inds)
    print(f"\nSAMLET: {status} — {msg}\n")
    for ind in inds:
        icon = {"GROEN": "✅", "GOEL": "⚠️", "ROED": "🔴", "GRAA": "⚫"}[ind["status"]]
        print(f"{icon} {ind['name']}: {ind['signal']}")

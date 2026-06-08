"""portfolio/signals.py

Beregn salgs-/hold-signaler for en portefoelge.

Signaler (prioriteret raekkefoelge):
  SAELG      — Death cross aktiv (SMA50 < SMA200)
  ADVARSEL   — Et eller flere af:
                 * Kurs under SMA50 (svaekkende trend)
                 * RSI > 75 (overkobt)
                 * 4-ugers afkast < -7%
  BEHOLD     — Alt ser fint ud
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class Signal:
    ticker: str
    name: str
    action: str  # "SAELG" | "ADVARSEL" | "BEHOLD"
    reasons: list[str]  # Forklaringer

    # Raadata
    close: float = float("nan")
    sma50: float = float("nan")
    sma200: float = float("nan")
    rsi: float = float("nan")
    return_4w: float = float("nan")
    dist_52w_high: float = float("nan")


def _rsi(close: pd.Series, period: int = 14) -> float:
    if len(close) < period + 1:
        return float("nan")
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    last_loss = avg_loss.iloc[-1]
    if last_loss == 0:
        return 100.0
    rs = avg_gain.iloc[-1] / last_loss
    return float(100 - 100 / (1 + rs))


def compute_signals(
    data: dict[str, pd.DataFrame],
    holdings: list,  # list[Holding]
    as_of: date | None = None,
) -> list[Signal]:
    """Beregn signal for hver aktie i holdings."""
    results: list[Signal] = []

    for holding in holdings:
        ticker = holding.ticker
        df = data.get(ticker)

        if df is None or df.empty:
            results.append(
                Signal(
                    ticker=ticker,
                    name=holding.name,
                    action="INGEN DATA",
                    reasons=["Ingen kursdata tilgængeligt"],
                )
            )
            continue

        df = df.sort_index()
        if as_of is not None:
            df = df[df.index <= pd.Timestamp(as_of)]
        if df.empty:
            results.append(
                Signal(
                    ticker=ticker,
                    name=holding.name,
                    action="INGEN DATA",
                    reasons=["Ingen data frem til dato"],
                )
            )
            continue

        close = df["close"].dropna()
        if len(close) < 20:
            results.append(
                Signal(
                    ticker=ticker,
                    name=holding.name,
                    action="INGEN DATA",
                    reasons=["For lidt historik"],
                )
            )
            continue

        last_close = float(close.iloc[-1])

        # SMA
        sma50_val = (
            float(close.rolling(50).mean().iloc[-1])
            if len(close) >= 50
            else float("nan")
        )
        sma200_val = (
            float(close.rolling(200).mean().iloc[-1])
            if len(close) >= 200
            else float("nan")
        )

        # RSI
        rsi_val = _rsi(close)

        # 4-ugers afkast
        ret_4w = (
            float(close.iloc[-1] / close.iloc[-20] - 1)
            if len(close) >= 20
            else float("nan")
        )

        # 52-ugers high
        high = df["high"].dropna() if "high" in df.columns else close
        w = min(252, len(high))
        high_52w = float(high.iloc[-w:].max())
        dist_52w = float(last_close / high_52w - 1) if high_52w > 0 else float("nan")

        # --- Signal-logik ---
        reasons: list[str] = []
        action = "BEHOLD"

        # SAELG: death cross
        if not pd.isna(sma50_val) and not pd.isna(sma200_val):
            if sma50_val < sma200_val:
                reasons.append(
                    f"Death cross: SMA50 ({sma50_val:.2f}) < SMA200 ({sma200_val:.2f})"
                )
                action = "SAELG"

        # ADVARSEL-tjek (gaelder ogsaa selvom action allerede er SAELG)
        warnings: list[str] = []

        if not pd.isna(sma50_val) and last_close < sma50_val:
            warnings.append(f"Kurs ({last_close:.2f}) under SMA50 ({sma50_val:.2f})")

        if not pd.isna(rsi_val) and rsi_val > 75:
            warnings.append(f"RSI overkobt: {rsi_val:.1f}")

        if not pd.isna(ret_4w) and ret_4w < -0.07:
            warnings.append(f"4-ugers afkast: {ret_4w:.1%}")

        if warnings:
            if action == "BEHOLD":
                action = "ADVARSEL"
            reasons.extend(warnings)

        if not reasons:
            reasons.append("Ingen negative signaler")

        results.append(
            Signal(
                ticker=ticker,
                name=holding.name,
                action=action,
                reasons=reasons,
                close=last_close,
                sma50=sma50_val,
                sma200=sma200_val,
                rsi=rsi_val,
                return_4w=ret_4w,
                dist_52w_high=dist_52w,
            )
        )

    # Sorter: SAELG -> ADVARSEL -> BEHOLD -> INGEN DATA
    order = {"SAELG": 0, "ADVARSEL": 1, "BEHOLD": 2, "INGEN DATA": 3}
    results.sort(key=lambda s: order.get(s.action, 9))
    return results

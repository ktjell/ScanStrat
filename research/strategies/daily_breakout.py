"""research/strategies/daily_breakout.py

DailyBreakout: estimerer sandsynlighed for at en aktie stiger >= TARGET% i dag.

Metode: historisk betinget hit-rate.
  For hver aktie beregnes: "givet at signalerne i gar lignede dem i dag,
  hvor tit steg aktien >= TARGET% den folgende dag?"

Signaler der bruges som betingelse:
  - RSI < RSI_MAX (oversold eller neutral)
  - Volume ratio >= VOL_RATIO_MIN (stigende interesse)
  - Aktie over sin 50d SMA (uptrend)
  - Foregaende dags afkast var positivt (impuls)
  - Naer 52-ugers hoj (breakout zone) ELLER naer 52-ugers lav (reversal)

Score = andel af dage med lignende signal-profil, hvor naeste dag >= TARGET%.
Minimum LOOKBACK_DAYS historiske dage kraeves for at give en score.

Kan bruges som dag-screening: "hvem har hoejest P(+3% i dag)?"
Backtesting giver lavere CAGR end momentum, men kan vaere nyttigt som
filter oven pa andre strategier.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datetime import date

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Parametre — tilpas her
# ------------------------------------------------------------------
TARGET_RETURN: float = 0.03  # vi vil have P(afkast >= 3% i morgen)
LOOKBACK_DAYS: int = 252  # historiske dage at beregne hit-rate over
MIN_MATCHES: int = 10  # minimum signalmatches for palidelig score
RSI_MAX: float = 60.0  # RSI skal vaere under dette (ikke overkoebt)
VOL_RATIO_MIN: float = 1.2  # volumen skal vaere over gennemsnit


def _compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    return 100 - 100 / (1 + rs)


def _score_ticker_vectorized(
    close: pd.Series,
    volume: pd.Series | None,
    target: float = TARGET_RETURN,
    lookback: int = LOOKBACK_DAYS,
    rsi_max: float = RSI_MAX,
    vol_ratio_min: float = VOL_RATIO_MIN,
    min_matches: int = MIN_MATCHES,
) -> pd.Series:
    """Vectorized rolling hit-rate for one ticker over its entire history.

    For hvert tidspunkt t: P(close[t+1]/close[t]-1 >= target | signaler[t]).
    Returnerer en dato-indekseret Series med NaN hvor der ikke er nok data.
    """
    rsi = _compute_rsi(close)
    sma50 = close.rolling(50).mean()

    if volume is not None:
        vol_ratio = volume / volume.rolling(20).mean()
    else:
        vol_ratio = pd.Series(1.0, index=close.index)

    # fwd_return[t] = return fra t til t+1 (ingen data-leak: vi predicter fremtiden)
    fwd_ret = close.pct_change().shift(-1)
    hit = (fwd_ret >= target).astype(float)

    sig_count = (
        (rsi < rsi_max).astype(int)
        + (vol_ratio >= vol_ratio_min).astype(int)
        + (close > sma50).astype(int)
    )
    match = (sig_count >= 2).astype(float)

    roll_match = match.rolling(lookback).sum()
    roll_hit = (hit * match).rolling(lookback).sum()

    # shift(1): score[t] maa kun bruge data op til t-1.
    # Uden shift ville roll_hit[t] inkludere hit[t] = afkast fra t til t+1
    # (fremtidens afkast) — det er look-ahead bias.
    return (roll_hit / roll_match).where(roll_match >= min_matches).shift(1)


def _score_ticker(df: pd.DataFrame, as_of_idx: int) -> float:
    """Beregn P(naeste_dags_afkast >= TARGET) baseret pa historisk betinget frekvens."""
    close = df["close"]
    volume = df["volume"] if "volume" in df.columns else None

    if len(close) < LOOKBACK_DAYS + 5:
        return float("nan")

    # Brug kun data op til (og med) as_of_idx
    c = close.iloc[: as_of_idx + 1]
    v = volume.iloc[: as_of_idx + 1] if volume is not None else None

    rsi = _compute_rsi(c).iloc[-LOOKBACK_DAYS:]
    sma50 = c.rolling(50).mean().iloc[-LOOKBACK_DAYS:]
    c_window = c.iloc[-LOOKBACK_DAYS:]

    vol_ratio = (
        (v / v.rolling(20).mean()).iloc[-LOOKBACK_DAYS:]
        if v is not None
        else pd.Series(1.0, index=c_window.index)
    )

    # Naeste dags afkast (shift -1 = fremadrettet)
    fwd_return = c_window.pct_change().shift(-1)

    # Signal-flags for historiske dage
    sig_rsi = rsi < RSI_MAX
    sig_vol = vol_ratio >= VOL_RATIO_MIN
    sig_trend = c_window > sma50

    # Kombiner: mindst 2 af 3 signaler skal vaere true (fleksibelt)
    signal_count = sig_rsi.astype(int) + sig_vol.astype(int) + sig_trend.astype(int)
    mask = (signal_count >= 2) & fwd_return.notna()

    matches = mask.sum()
    if matches < MIN_MATCHES:
        return float("nan")

    hit_rate = float((fwd_return[mask] >= TARGET_RETURN).mean())
    return hit_rate


class DailyBreakoutStrategy:
    """Ranger aktier efter historisk P(+TARGET% i morgen) baseret pa dagssignaler.

    Samme .rank(data, as_of) interface som de andre strategier.

    Backtest-optimering: ved foerste kald med et nyt data-dict precomputes hele
    score-matricen (dato x ticker) vektoriseret.  Alle efterfolgende kald er O(1)
    row-lookups, saa 1d-rebalancering over 5 aar tager sekunder, ikke timer.
    """

    def __init__(
        self,
        target_return: float = TARGET_RETURN,
        lookback_days: int = LOOKBACK_DAYS,
        rsi_max: float = RSI_MAX,
        vol_ratio_min: float = VOL_RATIO_MIN,
    ) -> None:
        self._target = target_return
        self._lookback = lookback_days
        self._rsi_max = rsi_max
        self._vol_ratio_min = VOL_RATIO_MIN
        self._matrix: pd.DataFrame | None = None
        self._data_id: int | None = None

    # ------------------------------------------------------------------
    # Precomputation
    # ------------------------------------------------------------------

    def _ensure_matrix(self, data: dict[str, pd.DataFrame]) -> None:
        """Byg score-matricen vektoriseret hvis data er nyt."""
        if self._matrix is not None and self._data_id == id(data):
            return

        self._data_id = id(data)
        series: dict[str, pd.Series] = {}
        for ticker, df in data.items():
            df = df.sort_index()
            if df.empty or "close" not in df.columns:
                continue
            close = df["close"]
            volume = df["volume"] if "volume" in df.columns else None
            s = _score_ticker_vectorized(
                close,
                volume,
                target=self._target,
                lookback=self._lookback,
                rsi_max=self._rsi_max,
                vol_ratio_min=self._vol_ratio_min,
            )
            s.index = pd.to_datetime(s.index).normalize()
            series[ticker] = s

        self._matrix = pd.DataFrame(series) if series else pd.DataFrame()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rank(
        self,
        data: dict[str, pd.DataFrame],
        as_of: date | None = None,
    ) -> pd.DataFrame:
        if not data:
            return pd.DataFrame()

        self._ensure_matrix(data)

        if self._matrix is None or self._matrix.empty:
            return pd.DataFrame()

        if as_of is None:
            ts = self._matrix.index.max()
        else:
            ts = pd.Timestamp(as_of)
            valid = self._matrix.index[self._matrix.index <= ts]
            if valid.empty:
                return pd.DataFrame()
            ts = valid[-1]

        row = self._matrix.loc[ts].dropna().sort_values(ascending=False)
        if row.empty:
            return pd.DataFrame()

        result = pd.DataFrame(
            {"ticker": row.index, "prob_breakout": row.values}
        ).reset_index(drop=True)
        result.index = result.index + 1
        result.index.name = "rank"
        return result.reset_index()


REBALANCE_DAYS: int = 1  # koeb i dag, saelg naeste handelsdag
TOP_N: int = 2  # de 1-2 aktier med hoejest sandsynlighed

# Position-stoerrelse: vi vil tjene 2000-3000 DKK ved 3% stigning
# Midpunkt 2500 DKK / 0.03 / 6.9 DKK per USD ≈ 12.077 USD per position
TARGET_PROFIT_DKK: float = 2_500.0
DKK_PER_USD: float = 6.9  # opdater ved stort kurs-skift
_position_usd = TARGET_PROFIT_DKK / TARGET_RETURN / DKK_PER_USD
PORTFOLIO_USD: float = _position_usd * TOP_N  # ca. 24.154 USD i alt


def build(
    rebalance_days: int = REBALANCE_DAYS,
    top_n: int = TOP_N,
    portfolio_usd: float = PORTFOLIO_USD,
) -> tuple[str, DailyBreakoutStrategy, int, int, float]:
    return (
        "Daily Breakout",
        DailyBreakoutStrategy(),
        rebalance_days,
        top_n,
        portfolio_usd,
    )

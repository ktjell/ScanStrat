"""research/strategies/golden_cross.py

Golden Cross / Death Cross strategi.

Logik:
  - KOB når SMA50 krydser OVER SMA200 (golden cross)
  - SAELG når SMA50 krydser UNDER SMA200 (death cross)

I praksis ved rebalancering:
  - Kun aktier med aktiv golden cross (SMA50 >= SMA200) er kandidater
  - Rangeret efter styrken af crosset: SMA50/SMA200 - 1
    (jo mere SMA50 er over SMA200, jo stærkere trend)

Vectoriseret: score-matricen beregnes ét slag ved foerste kald.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd

REBALANCE_DAYS: int = 5  # ugentlig — reagerer hurtigt paa nye crosses
MAX_POSITIONS: int = 15
MAX_NEW_BUYS: int = 10
SMA_FAST: int = 50
SMA_SLOW: int = 200
MIN_CROSS_PCT: float = 0.005
TRAILING_STOP_PCT: float = 0.08
SMA_EXIT: int = 20
# Markedsbredde-filter: faerre end BREADTH_THRESHOLD af universet i golden cross
# = bredt bear-marked → saat ALT i cash med det samme
BREADTH_THRESHOLD: float = 0.40
# Freshness-filter: koeb kun aktier hvor golden cross startede for max N dage siden.
# Undgaar at koebe ind i udtoemte trends (aktier der allerede er steget 200%).
# 0 = deaktiveret
MAX_CROSS_AGE: int = 90
# RSI-filter paa nye koeb: undgaar at koebe oversolgte OG overkobte aktier.
# RSI < RSI_MIN = momentum ikke startet endnu (vent)
# RSI > RSI_MAX = allerede overbought, forhoejet reversal-risiko
# 0/100 = deaktiveret
RSI_PERIOD: int = 14 # dage til RSI-beregning
RSI_MIN: float = 40.0
RSI_MAX: float = 65.0


class GoldenCrossStrategy:
    """Køb ved golden cross, sælg ved death cross.

    Returnerer kun aktier med aktiv golden cross (SMA50 >= SMA200),
    rangeret efter crossets styrke. Death cross = ikke med i output.
    """

    def __init__(
        self,
        sma_fast: int = SMA_FAST,
        sma_slow: int = SMA_SLOW,
        min_cross_pct: float = MIN_CROSS_PCT,
        max_positions: int = MAX_POSITIONS,
        max_new_buys: int = MAX_NEW_BUYS,
        trailing_stop_pct: float = TRAILING_STOP_PCT,
        sma_exit: int = SMA_EXIT,
        breadth_threshold: float = BREADTH_THRESHOLD,
        max_cross_age: int = MAX_CROSS_AGE,
        rsi_period: int = RSI_PERIOD,
        rsi_min: float = RSI_MIN,
        rsi_max: float = RSI_MAX,
    ) -> None:
        self._fast = sma_fast
        self._slow = sma_slow
        self._min_cross = min_cross_pct
        self.max_positions = max_positions
        self.max_new_buys = max_new_buys
        self._trailing_stop = trailing_stop_pct
        self._sma_exit = sma_exit
        self._breadth_threshold = breadth_threshold
        self._max_cross_age = max_cross_age
        self._rsi_period = rsi_period
        self._rsi_min = rsi_min
        self._rsi_max = rsi_max
        self._score_matrix: pd.DataFrame | None = None
        self._exit_matrix: pd.DataFrame | None = None
        self._rsi_matrix: pd.DataFrame | None = None
        self._breadth_series: pd.Series | None = None  # andel i golden cross per dag
        self._data_id: int | None = None

    def _ensure_matrix(self, data: dict[str, pd.DataFrame]) -> None:
        if self._score_matrix is not None and self._data_id == id(data):
            return

        self._data_id = id(data)
        scores: dict[str, pd.Series] = {}

        for ticker, df in data.items():
            df = df.sort_index()
            if "close" not in df.columns or df.empty:
                continue
            close = df["close"].dropna()
            if len(close) < self._slow + 5:
                continue

            close.index = pd.to_datetime(close.index).normalize()
            sma_fast = close.rolling(self._fast).mean()
            sma_slow = close.rolling(self._slow).mean()

            # Score = SMA50/SMA200 - 1
            # Positiv = golden cross, negativ = death cross
            score = sma_fast / sma_slow - 1
            scores[ticker] = score

        self._score_matrix = pd.DataFrame(scores) if scores else pd.DataFrame()

        # --- Exit-matrix: True = exit signal aktiv denne dag ---
        # Saelg naar kurs falder under SMA50 — MEGET hurtigere end at vente paa
        # at SMA50 krydser under SMA200 (death cross kan komme uger for sent).
        # "Pris under SMA50" = trenden er brudt, uanset om death cross er aktivt.
        exits: dict[str, pd.Series] = {}
        for ticker, df in data.items():
            df = df.sort_index()
            if "close" not in df.columns or df.empty:
                continue
            close = df["close"].dropna()
            if len(close) < self._slow + 5:
                continue
            close.index = pd.to_datetime(close.index).normalize()
            sma_fast = close.rolling(self._fast).mean()
            exits[ticker] = close < sma_fast  # pris under SMA50 = exit

        self._exit_matrix = pd.DataFrame(exits) if exits else pd.DataFrame()

        # --- RSI-matrice ---
        rsi_dict: dict[str, pd.Series] = {}
        for ticker, df in data.items():
            df = df.sort_index()
            if "close" not in df.columns or df.empty:
                continue
            close = df["close"].dropna()
            if len(close) < self._slow + 5:
                continue
            close.index = pd.to_datetime(close.index).normalize()
            delta = close.diff()
            gain = delta.clip(lower=0).rolling(self._rsi_period).mean()
            loss = (-delta.clip(upper=0)).rolling(self._rsi_period).mean()
            rs = gain / loss.replace(0, float("nan"))
            rsi_dict[ticker] = 100 - 100 / (1 + rs)

        self._rsi_matrix = pd.DataFrame(rsi_dict) if rsi_dict else pd.DataFrame()

        # --- Breadth-serie: andel af universet i golden cross per dag ---
        # score_matrix > 0 = golden cross. Tag mean (andel True) per dato.
        if not self._score_matrix.empty:
            self._breadth_series = (self._score_matrix > 0).mean(axis=1)
        else:
            self._breadth_series = pd.Series(dtype=float)

    def _days_since_cross(self, ticker: str, as_of: pd.Timestamp) -> int | None:
        """Antal dage siden det seneste golden cross startede for tickeren.

        Returnerer None hvis ticker ikke er i golden cross paa as_of-datoen.
        """
        if self._score_matrix is None or ticker not in self._score_matrix.columns:
            return None
        scores = self._score_matrix[ticker]
        scores = scores[scores.index <= as_of].dropna()
        if scores.empty or scores.iloc[-1] <= 0:
            return None  # ikke i golden cross
        in_gc = scores > 0
        # Find sidst dato hvor vi gik fra False → True (cross startede)
        transitions = in_gc & ~in_gc.shift(1).fillna(False)
        last_cross_dates = transitions[transitions].index
        if last_cross_dates.empty:
            return None
        cross_date = last_cross_dates[-1]
        return (as_of - cross_date).days

    def rank(
        self,
        data: dict[str, pd.DataFrame],
        as_of: date | None = None,
    ) -> pd.DataFrame:
        if not data:
            return pd.DataFrame()

        self._ensure_matrix(data)

        if self._score_matrix is None or self._score_matrix.empty:
            return pd.DataFrame()

        if as_of is None:
            ts = self._score_matrix.index.max()
        else:
            ts = pd.Timestamp(as_of)
            valid = self._score_matrix.index[self._score_matrix.index <= ts]
            if valid.empty:
                return pd.DataFrame()
            ts = valid[-1]

        row = self._score_matrix.loc[ts].dropna()

        # Kun golden cross med minimum buffer (score > min_cross_pct)
        row = row[row > self._min_cross].sort_values(ascending=False)

        if row.empty:
            return pd.DataFrame()

        result = pd.DataFrame({"ticker": row.index, "cross_score": row.values})
        result.index = range(1, len(result) + 1)
        result.index.name = "rank"
        return result.reset_index()

    def rebalance(
        self,
        current_tickers: list[str],
        data: dict[str, pd.DataFrame],
        as_of: "date | None" = None,
    ) -> list[str]:
        """Stateful rebalancering: hold gode, erstat svage med bedre.

        Logik:
          1. Beregn idealportefoelge = top-max_positions golden cross denne uge
          2. Behold eksisterende der stadig er i idealportefoelgen
          3. Saelg automatisk: death cross + dem der er gledet ud af idealportefoelgen
          4. Koeb nye ind (dem i idealportefoelgen vi ikke har endnu),
             maks max_new_buys per uge
        """
        ranked = self.rank(data, as_of=as_of)
        if ranked.empty:
            return []

        # --- Markedsbredde-filter ---
        # Hvis faerre end breadth_threshold af universet er i golden cross
        # er vi i et bredt bear-marked → saat alt i cash
        if self._breadth_series is not None and not self._breadth_series.empty:
            ts_b = pd.Timestamp(as_of) if as_of else self._breadth_series.index.max()
            valid_b = self._breadth_series.index[self._breadth_series.index <= ts_b]
            if len(valid_b) > 0:
                breadth = float(self._breadth_series.loc[valid_b[-1]])
                if breadth < self._breadth_threshold:
                    return []  # bear-regime: ingen positioner

        scores = dict(zip(ranked["ticker"], ranked["cross_score"]))

        # Find exit-signaler for denne dato
        forced_exits: set[str] = set()
        if self._exit_matrix is not None and not self._exit_matrix.empty:
            ts_exit = pd.Timestamp(as_of) if as_of else self._exit_matrix.index.max()
            valid = self._exit_matrix.index[self._exit_matrix.index <= ts_exit]
            if len(valid) > 0:
                exit_row = self._exit_matrix.loc[valid[-1]]
                forced_exits = set(exit_row[exit_row == True].index)

        # Idealporten denne uge: top-max_positions golden cross
        ideal = ranked["ticker"].tolist()[: self.max_positions]
        ideal_set = set(ideal)

        # Behold eksisterende der:
        #   - stadig er i idealporten (golden cross ok)
        #   - IKKE har exit-signal (trailing stop + under SMA20)
        kept = [t for t in current_tickers if t in ideal_set and t not in forced_exits]

        # Nye koeb = dem i idealporten vi ikke allerede har OG ikke har exit-signal
        # (undgaar at saelge og straks koebe den samme aktie igen)
        held_set = set(kept)
        new_needed = [t for t in ideal if t not in held_set and t not in forced_exits]

        # Freshness-filter: spring over aktier hvis cross er for gammelt (udtømt trend)
        if self._max_cross_age > 0:
            ts_age = pd.Timestamp(as_of) if as_of else self._score_matrix.index.max()
            new_needed = [
                t
                for t in new_needed
                if (age := self._days_since_cross(t, ts_age)) is not None
                and age <= self._max_cross_age
            ]

        # RSI-filter: koeb kun nye positioner med RSI i [rsi_min, rsi_max]
        # Eksisterende positioner beholdes uanset RSI (exit-logik klarer det)
        if self._rsi_matrix is not None and not self._rsi_matrix.empty:
            ts_rsi = pd.Timestamp(as_of) if as_of else self._rsi_matrix.index.max()
            valid_rsi = self._rsi_matrix.index[self._rsi_matrix.index <= ts_rsi]
            if len(valid_rsi) > 0:
                rsi_row = self._rsi_matrix.loc[valid_rsi[-1]]
                new_needed = [
                    t
                    for t in new_needed
                    if t in rsi_row.index
                    and self._rsi_min <= rsi_row[t] <= self._rsi_max
                ]

        new_buys = new_needed[: self.max_new_buys]

        return kept + new_buys


def build(
    rebalance_days: int = REBALANCE_DAYS,
    max_positions: int = MAX_POSITIONS,
    max_new_buys: int = MAX_NEW_BUYS,
    min_cross_pct: float = MIN_CROSS_PCT,
    trailing_stop_pct: float = TRAILING_STOP_PCT,
    sma_exit: int = SMA_EXIT,
    breadth_threshold: float = BREADTH_THRESHOLD,
    max_cross_age: int = MAX_CROSS_AGE,
    rsi_period: int = RSI_PERIOD,
    rsi_min: float = RSI_MIN,
    rsi_max: float = RSI_MAX,
) -> tuple:
    strat = GoldenCrossStrategy(
        min_cross_pct=min_cross_pct,
        max_positions=max_positions,
        max_new_buys=max_new_buys,
        trailing_stop_pct=trailing_stop_pct,
        sma_exit=sma_exit,
        breadth_threshold=breadth_threshold,
        max_cross_age=max_cross_age,
        rsi_period=rsi_period,
        rsi_min=rsi_min,
        rsi_max=rsi_max,
    )
    return "Golden Cross", strat, rebalance_days, max_positions

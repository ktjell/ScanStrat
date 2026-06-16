"""research/strategies/ml_ranker.py

ML Ranker: XGBoost walk-forward klassifikator.

Maal: ranger aktier efter P(slaar benchmark naeste uge).
Det er et klassifikations-problem — ikke en krystalkugle.

Walk-forward logik (INGEN look-ahead bias):
  Train paa 2020-2021 → predict 2022 Q1
  Train paa 2020-2022 Q1 → predict 2022 Q2
  ... osv., retrain hver RETRAIN_WEEKS uge

Features per aktie per uge (alt beregnet ud fra data t.o.m. as_of):
  Momentum:       ret_1w, ret_4w, ret_12w, ret_26w
  Teknik:         rsi_14, sma_ratio (SMA50/SMA200-1), dist_sma50 (close/SMA50-1)
  Volatilitet:    atr_norm (ATR14/close), realised_vol_20d
  Relativ styrke: ret_4w_vs_spy, ret_12w_vs_spy
  Marked:         spy_ret_4w, spy_rsi_14, spy_sma_ratio
  Regime:         is_bull (0/1 baseret paa SPY SMA200)
  Breadth:        pct_universe_above_sma50, pct_universe_golden_cross
  Volume:         vol_ratio (gennemsnitlig volume / 20d snit)

Label: 1 hvis aktiens naeste-uges afkast > SPY's naeste-uges afkast, else 0.
"""

from __future__ import annotations

import logging
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Parametre
# ------------------------------------------------------------------
REBALANCE_DAYS: int = 5
TOP_N: int = 15

# Walk-forward: minimum antal uger til foerste traeningssaet
MIN_TRAIN_WEEKS: int = 52  # ~1 aars data inden foerste prediction
RETRAIN_WEEKS: int = 4  # retrain model hver 4 uge
MIN_SAMPLES: int = 200  # minimum traeningseksempler
# Rolling window: traen kun paa de seneste N ugers data (None = expanding/al historik)
# ~104 uger (2 aar) er et godt udgangspunkt — adaptivt uden at vaere ustabilt
TRAIN_WINDOW_WEEKS: int | None = None

# Smart rebalancering: hold positioner i minimum N dage og skift kun
# ud naar en ny kandidat er markant bedre (undgaar overflødige handler)
MIN_HOLD_DAYS: int = 10  # minimum dage en position beholdes
SWAP_THRESHOLD: float = 0.10  # ny aktie skal vaere mindst 10%-point mere sandsynlig

# XGBoost hyperparametre — konservative for at undgaa overfitting
XGB_PARAMS: dict = {
    "n_estimators": 100,
    "max_depth": 3,  # lavt = lidt underfitting, undgaar overfitting
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,  # hoej = undgaar at fitte paa outliers
    "scale_pos_weight": 1.0,
    "eval_metric": "logloss",
    "random_state": 42,
    "verbosity": 0,
    "use_label_encoder": False,
}

# Navn paa SPY-ticker i data-dict — bruges til marked-features
# Sæt til benchmark-tickeren (hentes separat af runner og tilfoejes data-dict)
SPY_TICKER: str = "SPY"


class MLRankerStrategy:
    """XGBoost walk-forward ranker.

    Implementerer rank(data, as_of) — samme interface som de andre strategier.
    Modellen traenes/retraenes automatisk naar as_of nærmer sig kanten af
    det seneste traeningssaet.
    """

    def __init__(
        self,
        top_n: int = TOP_N,
        min_train_weeks: int = MIN_TRAIN_WEEKS,
        retrain_weeks: int = RETRAIN_WEEKS,
        min_samples: int = MIN_SAMPLES,
        xgb_params: dict | None = None,
        spy_ticker: str = SPY_TICKER,
        train_cutoff: date | None = None,
        train_window_weeks: int | None = TRAIN_WINDOW_WEEKS,
        smart_rebalance: bool = False,
        min_hold_days: int = MIN_HOLD_DAYS,
        swap_threshold: float = SWAP_THRESHOLD,
    ) -> None:
        self._top_n = top_n
        self._min_train_weeks = min_train_weeks
        self._retrain_weeks = retrain_weeks
        self._min_samples = min_samples
        self._xgb_params = xgb_params or XGB_PARAMS.copy()
        self._spy = spy_ticker
        # OOS-mode: hvis sat, traenes modellen KUN paa data <= train_cutoff,
        # og retraenes aldrig herefter. Alt efter train_cutoff er out-of-sample.
        self._train_cutoff: pd.Timestamp | None = (
            pd.Timestamp(train_cutoff) if train_cutoff is not None else None
        )
        # Rolling window: kun de seneste N uger bruges til traeningsdata
        # None = expanding window (al historik)
        self._train_window: pd.Timedelta | None = (
            pd.Timedelta(weeks=train_window_weeks) if train_window_weeks else None
        )
        # Smart rebalancering
        self._smart_rebalance = smart_rebalance
        self._min_hold_days = min_hold_days
        self._swap_threshold = swap_threshold
        # Stateful: hold-datoer per ticker (hvornaar blev de købt?)
        self._entry_dates: dict[str, pd.Timestamp] = {}

        self._model: XGBClassifier | None = None
        self._feature_cols: list[str] = []
        self._last_trained: pd.Timestamp | None = (
            None  # hvornaar model sidst blev traenet
        )
        self._retrain_interval = pd.Timedelta(weeks=retrain_weeks)

        # Cache: hele feature-matricen beregnes én gang per data-objekt
        self._feature_store: pd.DataFrame | None = None  # MultiIndex (date, ticker)
        self._labels_store: pd.Series | None = None  # 1 = slog SPY, 0 = slog ikke
        self._data_id: int | None = None

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, float("nan"))
        return 100 - 100 / (1 + rs)

    @staticmethod
    def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average True Range normaliseret til closing price."""
        if "high" not in df.columns or "low" not in df.columns:
            return pd.Series(dtype=float, index=df.index)
        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr.rolling(period).mean() / close

    def _compute_spy_features(self, spy_close: pd.Series) -> pd.DataFrame:
        """Beregn markedsbredte features ud fra SPY."""
        spy_close = spy_close.sort_index()
        # Sikr tz-naive index saa reindex() matcher ticker-data fra parquet cache
        if spy_close.index.tz is not None:
            spy_close.index = spy_close.index.tz_localize(None)
        sma50 = spy_close.rolling(50).mean()
        sma200 = spy_close.rolling(200).mean()
        return pd.DataFrame(
            {
                "spy_ret_4w": spy_close.pct_change(20, fill_method=None),
                "spy_rsi_14": self._rsi(spy_close, 14),
                "spy_sma_ratio": sma50 / sma200 - 1,
                "is_bull": (spy_close >= sma200).astype(int),
            }
        )

    def _build_feature_store(self, data: dict[str, pd.DataFrame]) -> None:
        """Byg hele feature-matricen (date × ticker) paa én gang.

        Lagres som self._feature_store med MultiIndex (date, ticker).
        Inkluderer ogsaa labels (naeste-uges afkast vs SPY).
        """
        if self._data_id == id(data) and self._feature_store is not None:
            return

        self._data_id = id(data)

        # --- SPY market features ---
        # SPY er typisk ikke i universet — hent den direkte fra yfinance hvis nødvendig
        spy_df = data.get(self._spy)
        if spy_df is None or spy_df.empty or "close" not in spy_df.columns:
            try:
                import yfinance as yf

                # Find dato-range fra data
                all_dates = [
                    pd.to_datetime(df.index).normalize()
                    for df in data.values()
                    if not df.empty and len(df) > 0
                ]
                if all_dates:
                    min_date = min(d.min() for d in all_dates)
                    max_date = max(d.max() for d in all_dates)
                    raw = yf.download(
                        self._spy,
                        start=min_date - pd.Timedelta(days=10),
                        end=max_date + pd.Timedelta(days=5),
                        auto_adjust=True,
                        progress=False,
                    )
                    if not raw.empty:
                        # Håndter MultiIndex (yfinance >=0.2)
                        if isinstance(raw.columns, pd.MultiIndex):
                            raw.columns = [c[0].lower() for c in raw.columns]
                        else:
                            raw.columns = [c.lower() for c in raw.columns]
                        spy_df = raw
                        logger.info(
                            "SPY hentet direkte fra yfinance (%d rækker)", len(spy_df)
                        )
            except Exception as e:
                logger.warning("Kunne ikke hente SPY fra yfinance: %s", e)

        if spy_df is None or spy_df.empty or "close" not in spy_df.columns:
            logger.warning("SPY data ikke fundet — marked-features udelades")
            spy_features = pd.DataFrame()
            spy_fwd = pd.Series(dtype=float)
        else:
            spy_close = spy_df["close"].sort_index()
            spy_close.index = pd.to_datetime(spy_close.index).normalize()
            if spy_close.index.tz is not None:
                spy_close.index = spy_close.index.tz_localize(None)
            spy_features = self._compute_spy_features(spy_close)
            spy_fwd = spy_close.pct_change(5, fill_method=None).shift(
                -5
            )  # SPY's naeste-uges afkast

        # --- Breadth features: pct over SMA50 og i golden cross ---
        # Vigtigt: brug NaN hvor SMA endnu ikke er beregneligt (for lidt historik).
        # mean(axis=1) springer NaN over automatisk — saa breadth paa tidlige datoer
        # kun tæller tickers der faktisk havde nok historik paa den dato.
        # Undgaar at hele det nuvaerende univers paavirker breadth for 2020-datoer
        # hvor mange tickers ikke eksisterede eller manglede data.
        sma50_above: dict[str, pd.Series] = {}
        golden_cross: dict[str, pd.Series] = {}
        for ticker, df in data.items():
            if "close" not in df.columns or df.empty:
                continue
            close = df["close"].sort_index()
            close.index = pd.to_datetime(close.index).normalize()
            if len(close) < 205:
                continue
            sma50 = close.rolling(50).mean()
            sma200 = close.rolling(200).mean()
            # NaN hvor SMA endnu ikke er klar — mean() vil automatisk ekskludere dem
            valid50 = sma50.notna()
            valid200 = sma200.notna()
            sma50_above[ticker] = (close >= sma50).where(valid50).astype(float)
            golden_cross[ticker] = (
                (sma50 >= sma200).where(valid50 & valid200).astype(float)
            )

        if sma50_above:
            breadth_sma50 = pd.DataFrame(sma50_above).mean(axis=1)
            breadth_gc = pd.DataFrame(golden_cross).mean(axis=1)
        else:
            breadth_sma50 = pd.Series(dtype=float)
            breadth_gc = pd.Series(dtype=float)

        # --- Per-ticker features ---
        all_rows: list[pd.DataFrame] = []

        for ticker, df in data.items():
            df = df.sort_index().copy()
            if "close" not in df.columns or df.empty:
                continue
            close = df["close"].dropna()
            close.index = pd.to_datetime(close.index).normalize()
            if len(close) < 210:  # minimum for SMA200 + lidt buffer
                continue

            sma50 = close.rolling(50).mean()
            sma200 = close.rolling(200).mean()
            rsi14 = self._rsi(close, 14)
            vol20 = close.pct_change().rolling(20).std()

            has_vol = "volume" in df.columns
            if has_vol:
                volume = df["volume"].reindex(close.index)
                vol_ratio = volume / volume.rolling(20).mean()
            else:
                vol_ratio = pd.Series(np.nan, index=close.index)

            has_ohlc = "high" in df.columns and "low" in df.columns
            if has_ohlc:
                atr_norm = self._atr(df.reindex(close.index), 14)
            else:
                atr_norm = pd.Series(np.nan, index=close.index)

            # Fremadrettet label: slog aktien SPY naeste uge? (no look-ahead — shift(-5))
            fwd_ret = close.pct_change(5).shift(-5)
            if not spy_fwd.empty:
                spy_aligned = spy_fwd.reindex(close.index, method="ffill")
                label = (fwd_ret > spy_aligned).astype(float)
            else:
                label = (fwd_ret > 0).astype(float)  # fallback

            feat = pd.DataFrame(
                {
                    # Momentum
                    "ret_1w": close.pct_change(5),
                    "ret_4w": close.pct_change(20),
                    "ret_12w": close.pct_change(60),
                    "ret_26w": close.pct_change(130),
                    # Teknisk
                    "rsi_14": rsi14,
                    "sma_ratio": sma50 / sma200 - 1,
                    "dist_sma50": close / sma50 - 1,
                    # Volatilitet
                    "atr_norm": atr_norm,
                    "realised_vol_20d": vol20,
                    # Volumen
                    "vol_ratio": vol_ratio,
                    # Label
                    "_label": label,
                    "_ticker": ticker,
                },
                index=close.index,
            )

            # Tilfoej marked-features
            if not spy_features.empty:
                for col in spy_features.columns:
                    feat[col] = spy_features[col].reindex(close.index, method="ffill")

            # Tilfoej relativ styrke vs SPY
            if not spy_fwd.empty or not spy_features.empty:
                spy_ret4w = (
                    spy_features["spy_ret_4w"].reindex(close.index, method="ffill")
                    if "spy_ret_4w" in spy_features.columns
                    else 0
                )
                spy_ret12w_approx = close.pct_change(
                    60
                )  # vi mangler spy 12w direkte, approx med spy_ret_4w * 3
                feat["ret_4w_vs_spy"] = feat["ret_4w"] - spy_ret4w
                feat["ret_12w_vs_spy"] = feat["ret_12w"] - spy_ret4w * 3

            # Breadth
            if not breadth_sma50.empty:
                feat["breadth_sma50"] = breadth_sma50.reindex(
                    close.index, method="ffill"
                )
                feat["breadth_gc"] = breadth_gc.reindex(close.index, method="ffill")

            all_rows.append(feat)

        if not all_rows:
            self._feature_store = pd.DataFrame()
            self._labels_store = pd.Series(dtype=float)
            return

        combined = pd.concat(all_rows, ignore_index=False)
        combined = combined.reset_index().rename(columns={"index": "date"})
        combined["date"] = pd.to_datetime(combined["date"])
        combined = combined.set_index(["date", "_ticker"])

        self._labels_store = combined["_label"].rename("label")
        feature_cols = [c for c in combined.columns if not c.startswith("_")]
        self._feature_store = combined[feature_cols]
        self._feature_cols = feature_cols

        logger.debug(
            "Feature store bygget: %d rækker, %d features, %d tickers",
            len(self._feature_store),
            len(self._feature_cols),
            len(all_rows),
        )

    # ------------------------------------------------------------------
    # Traenings-logik
    # ------------------------------------------------------------------

    def _should_retrain(self, as_of: pd.Timestamp) -> bool:
        if self._model is None:
            return True
        if self._last_trained is None:
            return True
        # OOS-mode: model er fryst efter train_cutoff — aldrig retrain
        if self._train_cutoff is not None and as_of > self._train_cutoff:
            return False
        return (as_of - self._last_trained) >= self._retrain_interval

    def _train(self, as_of: pd.Timestamp) -> bool:
        """Traen model paa alle data STRIKT foer as_of. Returnerer True ved succes."""
        if self._feature_store is None or self._feature_store.empty:
            return False

        # Kun rækker med dato < as_of - 7 dage
        # (labels for de sidste 5 dage refererer til fremtidige priser — undgaa look-ahead)
        cutoff = as_of - pd.Timedelta(days=7)
        # OOS-mode: traen aldrig paa data efter train_cutoff
        if self._train_cutoff is not None:
            cutoff = min(cutoff, self._train_cutoff)
        dates_in_store = self._feature_store.index.get_level_values("date")
        mask = dates_in_store < cutoff

        # Rolling window: skær gamle data fra
        if self._train_window is not None:
            window_start = cutoff - self._train_window
            mask = mask & (dates_in_store >= window_start)

        X = self._feature_store.loc[mask].copy()
        y = self._labels_store.loc[mask].copy()

        # Fjern rækker med NaN labels eller features
        valid = y.notna() & X.notna().all(axis=1)
        X = X.loc[valid]
        y = y.loc[valid]

        if len(X) < self._min_samples:
            logger.debug(
                "For faa traeningseksempler (%d < %d) paa %s",
                len(X),
                self._min_samples,
                as_of.date(),
            )
            return False

        # Minimum MIN_TRAIN_WEEKS ugers data
        earliest = dates_in_store[mask].min()
        weeks_of_data = (as_of - earliest).days / 7
        if weeks_of_data < self._min_train_weeks:
            return False

        model = XGBClassifier(**self._xgb_params)
        model.fit(X.values, y.values.astype(int))

        self._model = model
        self._last_trained = as_of
        logger.info(
            "Model traenet paa %d eksempler (t.o.m. %s)",
            len(X),
            (as_of - timedelta(days=1)).date(),
        )
        return True

    # ------------------------------------------------------------------
    # rank() — offentlig interface
    # ------------------------------------------------------------------

    def rank(
        self,
        data: dict[str, pd.DataFrame],
        as_of: date | None = None,
    ) -> pd.DataFrame:
        """Rangér aktier efter P(slår SPY naeste uge).

        Returnerer DataFrame med kolonner [rank, ticker, ml_score].
        Tomme DataFrame = model ikke klar endnu (for lidt traeningsdata).
        """
        if not data:
            return pd.DataFrame()

        self._build_feature_store(data)

        if self._feature_store is None or self._feature_store.empty:
            return pd.DataFrame()

        if as_of is None:
            ts = self._feature_store.index.get_level_values("date").max()
        else:
            ts = pd.Timestamp(as_of)

        # Retrain hvis det er tid
        if self._should_retrain(ts):
            ok = self._train(ts)
            if not ok:
                return pd.DataFrame()  # for lidt data endnu

        if self._model is None:
            return pd.DataFrame()

        # Hent features for as_of (seneste tilgaengelige dato <= as_of)
        dates = self._feature_store.index.get_level_values("date")
        valid_dates = dates[dates <= ts]
        if valid_dates.empty:
            return pd.DataFrame()
        snap_date = valid_dates.max()

        try:
            snap = self._feature_store.xs(snap_date, level="date")
        except KeyError:
            return pd.DataFrame()

        # Drop rækker med NaN
        snap_clean = snap.dropna()
        if snap_clean.empty:
            return pd.DataFrame()

        proba = self._model.predict_proba(snap_clean.values)[:, 1]

        result = (
            pd.DataFrame(
                {
                    "ticker": snap_clean.index,
                    "ml_score": proba,
                }
            )
            .sort_values("ml_score", ascending=False)
            .reset_index(drop=True)
        )

        result.index = range(1, len(result) + 1)
        result.index.name = "rank"
        return result.reset_index()

    def rebalance(
        self,
        current_tickers: list[str],
        data: dict[str, pd.DataFrame],
        as_of: "date | None" = None,
    ) -> list[str]:
        """Stateful smart-rebalancering med minimum hold-tid og swap-threshold.

        Logik:
          1. Rangér alle aktier med ML-modellen
          2. EXIT altid (uanset hold-tid) hvis score < 0.40 — modellen er imod positionen
          3. BEHOLD hvis score >= 0.40 OG (holdt < min_hold_days ELLER score stadig ok)
          4. KØB nye kun hvis kandidat er swap_threshold bedre end svageste nuværende
             (anti-churn: undgaar at swipe ind og ud af lignende aktier)
        """
        ranked = self.rank(data, as_of=as_of)
        if ranked.empty:
            return current_tickers  # model ikke klar endnu — behold status quo

        ts = pd.Timestamp(as_of) if as_of else pd.Timestamp.now()
        scores = dict(zip(ranked["ticker"], ranked["ml_score"]))

        # Opdater entry-datoer for nye positioner
        for t in current_tickers:
            if t not in self._entry_dates:
                self._entry_dates[t] = ts

        # --- Trin 1: Forced exits (uanset hold-tid) ---
        # Score < 0.40 = modellen mener aktien har < 40% chance for at slå SPY
        # Det er et klart negativt signal — sælg altid, selv dag 1
        EXIT_SCORE = 0.40
        survivors = [t for t in current_tickers if scores.get(t, 0.0) >= EXIT_SCORE]
        forced_exits = set(current_tickers) - set(survivors)
        if forced_exits:
            for t in forced_exits:
                self._entry_dates.pop(t, None)

        # --- Trin 2: Opdel survivors i locked/unlocked ---
        # locked = holdt kortere end min_hold_days → beskyttes mod swap-churn
        # (men kun mod udskiftning med ny kandidat, IKKE mod forced exit ovenfor)
        locked: set[str] = set()
        unlocked: set[str] = set()
        for t in survivors:
            entry = self._entry_dates.get(t, ts)
            days_held = (ts - entry).days
            if days_held < self._min_hold_days:
                locked.add(t)
            else:
                unlocked.add(t)

        # --- Trin 3: Fyld op med nye kandidater ---
        ideal_top = ranked["ticker"].tolist()[: self._top_n]
        held_set = set(survivors)
        candidates = [t for t in ideal_top if t not in held_set]

        result = list(survivors)

        # Fyld direkte op hvis under max
        for candidate in candidates:
            if len(result) >= self._top_n:
                break
            result.append(candidate)
            self._entry_dates[candidate] = ts

        # --- Trin 4: Swap — udskift svageste unlocked med markant bedre kandidat ---
        for candidate in candidates:
            if candidate in set(result):
                continue
            candidate_score = scores.get(candidate, 0.0)
            unlocked_in_result = [t for t in result if t in unlocked]
            if not unlocked_in_result:
                break
            weakest = min(unlocked_in_result, key=lambda t: scores.get(t, 0.0))
            weakest_score = scores.get(weakest, 0.0)
            if candidate_score - weakest_score >= self._swap_threshold:
                result.remove(weakest)
                self._entry_dates.pop(weakest, None)
                unlocked.discard(weakest)
                result.append(candidate)
                self._entry_dates[candidate] = ts
            else:
                break  # ingen kandidat er god nok — stop

        # Ryd entry_dates for solgte positioner
        result_set = set(result)
        self._entry_dates = {
            t: d for t, d in self._entry_dates.items() if t in result_set
        }

        return result[: self._top_n]


def build(
    top_n: int = TOP_N,
    rebalance_days: int = REBALANCE_DAYS,
    min_train_weeks: int = MIN_TRAIN_WEEKS,
    retrain_weeks: int = RETRAIN_WEEKS,
    train_cutoff: date | None = None,
    train_window_weeks: int | None = TRAIN_WINDOW_WEEKS,
    smart_rebalance: bool = False,
    min_hold_days: int = MIN_HOLD_DAYS,
    swap_threshold: float = SWAP_THRESHOLD,
) -> tuple:
    parts = []
    if train_cutoff is not None:
        parts.append(f"OOS {train_cutoff.year + 1}+")
    if train_window_weeks is not None:
        parts.append(f"{train_window_weeks}w window")
    if smart_rebalance:
        parts.append(f"smart ({min_hold_days}d/{int(swap_threshold * 100)}%)")
    suffix = f" ({', '.join(parts)})" if parts else ""
    name = f"ML Ranker{suffix}"
    strat = MLRankerStrategy(
        top_n=top_n,
        min_train_weeks=min_train_weeks,
        retrain_weeks=retrain_weeks,
        train_cutoff=train_cutoff,
        train_window_weeks=train_window_weeks,
        smart_rebalance=smart_rebalance,
        min_hold_days=min_hold_days,
        swap_threshold=swap_threshold,
    )
    return name, strat, rebalance_days, top_n

"""research/strategies/ensemble_ranker.py

Multi-faktor Ensemble Ranker.

Tre specialiserede XGBoost-modeller trænes uafhængigt på hvert sit feature-subset:
  - Model A (Momentum):   ret_1w, ret_4w, ret_12w, ret_26w, ret_4w_vs_spy, ret_12w_vs_spy
  - Model B (Teknisk):    rsi_14, sma_ratio, dist_sma50, atr_norm, realised_vol_20d, vol_ratio
  - Model C (Makro):      spy_ret_4w, spy_rsi_14, spy_sma_ratio, is_bull, breadth_sma50, breadth_gc

Endelig score = vægtet gennemsnit af de tre modellers P(slår SPY næste uge).

Walk-forward logik er identisk med MLRankerStrategy (ingen look-ahead bias).
Interface er identisk — drop-in replacement i scorer og backtest.
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

# Genbrug feature-engineering fra MLRankerStrategy
from research.strategies.ml_ranker import MLRankerStrategy, XGB_PARAMS

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Parametre
# ------------------------------------------------------------------
REBALANCE_DAYS: int = 5
TOP_N: int = 15
MIN_TRAIN_WEEKS: int = 52
RETRAIN_WEEKS: int = 4
MIN_SAMPLES: int = 200

# Feature-subsets for de tre specialmodeller
MOMENTUM_FEATURES = [
    "ret_1w",
    "ret_4w",
    "ret_12w",
    "ret_26w",
    "ret_4w_vs_spy",
    "ret_12w_vs_spy",
]
TECHNICAL_FEATURES = [
    "rsi_14",
    "sma_ratio",
    "dist_sma50",
    "atr_norm",
    "realised_vol_20d",
    "vol_ratio",
]
MACRO_FEATURES = [
    "spy_ret_4w",
    "spy_rsi_14",
    "spy_sma_ratio",
    "is_bull",
    "breadth_sma50",
    "breadth_gc",
]

# Vægte for de tre modeller (summer til 1.0)
MODEL_WEIGHTS = {
    "momentum": 0.40,
    "technical": 0.35,
    "macro": 0.25,
}


class EnsembleRankerStrategy:
    """
    Multi-faktor ensemble af tre XGBoost-modeller.

    Implementerer rank(data, as_of) — samme interface som MLRankerStrategy.
    """

    def __init__(
        self,
        top_n: int = TOP_N,
        min_train_weeks: int = MIN_TRAIN_WEEKS,
        retrain_weeks: int = RETRAIN_WEEKS,
        min_samples: int = MIN_SAMPLES,
        xgb_params: dict | None = None,
        spy_ticker: str = "SPY",
        train_cutoff: date | None = None,
        weights: dict | None = None,
    ) -> None:
        self._top_n = top_n
        self._min_train_weeks = min_train_weeks
        self._retrain_weeks = retrain_weeks
        self._min_samples = min_samples
        self._xgb_params = xgb_params or XGB_PARAMS.copy()
        self._spy = spy_ticker
        self._train_cutoff: pd.Timestamp | None = (
            pd.Timestamp(train_cutoff) if train_cutoff else None
        )
        self._weights = weights or MODEL_WEIGHTS
        self._retrain_interval = pd.Timedelta(weeks=retrain_weeks)

        # Tre modeller
        self._models: dict[str, XGBClassifier | None] = {
            "momentum": None,
            "technical": None,
            "macro": None,
        }
        self._feature_subsets = {
            "momentum": MOMENTUM_FEATURES,
            "technical": TECHNICAL_FEATURES,
            "macro": MACRO_FEATURES,
        }

        # Deles med MLRankerStrategy til feature-engineering
        self._base = MLRankerStrategy(
            top_n=top_n,
            min_train_weeks=min_train_weeks,
            retrain_weeks=retrain_weeks,
            min_samples=min_samples,
            xgb_params=xgb_params or XGB_PARAMS.copy(),
            spy_ticker=spy_ticker,
            train_cutoff=train_cutoff,
        )

        self._last_trained: pd.Timestamp | None = None
        self._feature_cols: list[str] = []

    # ------------------------------------------------------------------
    # Public interface (identisk med MLRankerStrategy)
    # ------------------------------------------------------------------

    def rank(self, data: dict[str, pd.DataFrame], as_of: date) -> pd.DataFrame:
        """
        Returnerer DataFrame med kolonner: ticker, ensemble_score, rank
        samt individuelle model-scores for transparens.
        """
        as_of_ts = pd.Timestamp(as_of)

        # Byg feature store via base-klassen
        self._base._build_feature_store(data)
        if self._base._feature_store is None or self._base._feature_store.empty:
            logger.warning("Ingen feature data — kan ikke ranke")
            return pd.DataFrame()

        # Retrain hvis nødvendigt
        if self._should_retrain(as_of_ts):
            success = self._train(as_of_ts)
            if not success:
                logger.warning("Ensemble træning fejlede — for lidt data?")
                return pd.DataFrame()
            self._last_trained = as_of_ts

        # Predict på seneste tilgængelige dato per ticker
        store = self._base._feature_store
        labels = self._base._labels_store

        # Samme logik som MLRankerStrategy.rank(): seneste dato <= as_of
        dates = store.index.get_level_values("date")
        mask = dates <= as_of_ts
        if not mask.any():
            return pd.DataFrame()

        recent = store.loc[mask].groupby(level="_ticker").last()
        if recent.empty:
            return pd.DataFrame()

        # Predict med alle tre modeller
        scores = {}
        for name, model in self._models.items():
            if model is None:
                continue
            cols = [c for c in self._feature_subsets[name] if c in recent.columns]
            if not cols:
                continue
            X = recent[cols].copy()
            # Fyld NaN med median (robust til manglende features)
            X = X.fillna(X.median())
            valid = X.notna().all(axis=1)
            if not valid.any():
                continue
            proba = model.predict_proba(X.loc[valid])[:, 1]
            scores[name] = pd.Series(proba, index=X.loc[valid].index)

        if not scores:
            return pd.DataFrame()

        # Kombiner med vægte
        score_df = pd.DataFrame(scores)
        ensemble = sum(
            score_df[name] * self._weights.get(name, 1 / len(scores))
            for name in score_df.columns
        )

        # Byg output DataFrame
        result = pd.DataFrame(
            {
                "ticker": ensemble.index,
                "ensemble_score": ensemble.values,
            }
        )

        # Tilføj individuelle model-scores
        for name in ["momentum", "technical", "macro"]:
            if name in score_df.columns:
                result[f"{name}_score"] = score_df[name].reindex(ensemble.index).values
            else:
                result[f"{name}_score"] = np.nan

        # ml_score alias (til bagudkompatibilitet med scorer.py)
        result["ml_score"] = result["ensemble_score"]

        result = result.sort_values("ensemble_score", ascending=False).reset_index(
            drop=True
        )
        result["rank"] = result.index + 1
        return result

    # ------------------------------------------------------------------
    # Træning
    # ------------------------------------------------------------------

    def _should_retrain(self, as_of: pd.Timestamp) -> bool:
        if any(m is None for m in self._models.values()):
            return True
        if self._last_trained is None:
            return True
        if self._train_cutoff is not None and as_of > self._train_cutoff:
            return False
        return (as_of - self._last_trained) >= self._retrain_interval

    def _train(self, as_of: pd.Timestamp) -> bool:
        """Træn alle tre modeller på data strikt før as_of."""
        store = self._base._feature_store
        labels = self._base._labels_store

        if store is None or store.empty:
            return False

        cutoff = as_of - pd.Timedelta(days=7)
        if self._train_cutoff is not None:
            cutoff = min(cutoff, self._train_cutoff)

        dates = store.index.get_level_values("date")
        mask = dates < cutoff

        # Minimum træningslængde
        min_date = as_of - pd.Timedelta(weeks=self._min_train_weeks)
        if dates[mask].min() > min_date:
            logger.debug("For lidt historik til ensemble-træning endnu")
            return False

        X_all = store.loc[mask].copy()
        y_all = labels.loc[mask].copy()
        valid = y_all.notna() & X_all.notna().any(axis=1)
        X_all = X_all.loc[valid]
        y_all = y_all.loc[valid]

        if len(y_all) < self._min_samples:
            logger.debug("For få samples: %d < %d", len(y_all), self._min_samples)
            return False

        success_count = 0
        for name, feature_list in self._feature_subsets.items():
            cols = [c for c in feature_list if c in X_all.columns]
            if len(cols) < 2:
                logger.warning(
                    "Model '%s': kun %d features tilgængelige — springer over",
                    name,
                    len(cols),
                )
                continue

            X = X_all[cols].copy()
            y = y_all.copy()

            # Fjern rækker med NaN i disse specifikke features
            row_valid = X.notna().all(axis=1) & y.notna()
            X = X.loc[row_valid]
            y = y.loc[row_valid]

            if len(y) < self._min_samples:
                logger.debug("Model '%s': for få gyldige samples (%d)", name, len(y))
                continue

            model = XGBClassifier(**self._xgb_params)
            model.fit(X, y)
            self._models[name] = model
            success_count += 1
            logger.info(
                "Ensemble model '%s' trænet: %d samples, %d features",
                name,
                len(y),
                len(cols),
            )

        return success_count >= 2  # Mindst 2 af 3 modeller skal lykkes


def build(
    rebalance_days: int = REBALANCE_DAYS,
) -> tuple[str, "EnsembleRankerStrategy", int, int]:
    """Returnerer (navn, strategi-instans, rebalance_dage, top_n) klar til BacktestRunner."""
    return "Ensemble Ranker", EnsembleRankerStrategy(top_n=TOP_N), rebalance_days, TOP_N

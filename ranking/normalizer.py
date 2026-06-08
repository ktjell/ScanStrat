from __future__ import annotations

import pandas as pd


class Normalizer:
    """
    Converts raw feature values to 0–100 cross-sectional percentile scores.

    Each feature column is ranked against all tickers in the universe for
    that snapshot.  A ticker at the 87th percentile gets a score of 87.

    Cross-sectional percentile ranking is:
    - Scale-invariant: momentum (≈0.2) and SMA200 (≈150) are treated equally.
    - Outlier-robust: one extreme value does not distort the rest.
    - Requires no manual calibration.

    Parameters
    ----------
    ascending : dict[str, bool]
        Per-feature sort direction.  True (default) = higher raw value is
        better (e.g. momentum).  False = lower raw value is better
        (e.g. volatility — set to False so low vol → high score).
    """

    _DEFAULT_ASCENDING: dict[str, bool] = {
        "momentum_3m": True,
        "momentum_6m": True,
        "momentum_12m": True,
        "sma_50": True,
        "sma_200": True,
        "dist_52w_high": True,
        "death_cross": False,  # 0.0 (golden cross) → high score
        "rsi_14": True,
        "volatility_30d": False,  # low volatility → high score
    }

    def __init__(self, ascending: dict[str, bool] | None = None) -> None:
        self._ascending = (
            ascending if ascending is not None else dict(self._DEFAULT_ASCENDING)
        )

    def normalize(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Return a DataFrame of the same shape with values in [0, 100].

        Columns not listed in *ascending* are treated as ascending=True.
        NaN values are propagated (a ticker with NaN for a feature gets
        NaN for that feature's score — the scorer handles this).
        """
        scores = pd.DataFrame(index=features.index)
        for col in features.columns:
            ascending = self._ascending.get(col, True)
            scores[col] = (
                features[col].rank(
                    method="average", ascending=ascending, pct=True, na_option="keep"
                )
                * 100
            )
        return scores

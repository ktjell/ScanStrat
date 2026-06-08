from __future__ import annotations

import pandas as pd


class Scorer:
    """
    Computes a weighted composite score from normalised (0–100) feature scores.

    Weights are read from ``RankingSettings.weights`` and must sum to
    approximately 1.0 (they are re-normalised internally so the final score
    remains on a 0–100 scale regardless of what the weights actually sum to).

    Features absent from *weights* are ignored.
    NaN scores for a feature are treated as 50 (neutral) so that a single
    missing indicator does not disqualify an otherwise strong ticker.
    """

    _NAN_FILL: float = 50.0

    def __init__(self, weights: dict[str, float]) -> None:
        if not weights:
            raise ValueError("weights must not be empty")
        total = sum(abs(w) for w in weights.values())
        self._weights: dict[str, float] = {k: v / total for k, v in weights.items()}

    def score(self, normalised: pd.DataFrame) -> pd.Series:
        """
        Return a Series of composite scores indexed by ticker.

        Parameters
        ----------
        normalised : pd.DataFrame
            Output of ``Normalizer.normalize()`` — values in [0, 100],
            index = ticker, columns = feature names.

        Returns
        -------
        pd.Series
            Composite score in [0, 100], name="score".
        """
        result = pd.Series(0.0, index=normalised.index, name="score")
        for feature, weight in self._weights.items():
            if feature not in normalised.columns:
                continue
            col = normalised[feature].fillna(self._NAN_FILL)
            result += weight * col
        return result

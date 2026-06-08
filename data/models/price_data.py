from __future__ import annotations

import pandas as pd

PRICE_COLUMNS: list[str] = ["open", "high", "low", "close", "volume"]


def validate_price_df(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Ensure a DataFrame conforms to the canonical price schema.

    Expected schema
    ---------------
    Index   : DatetimeIndex (UTC-naive, daily frequency), name="date"
    Columns : open, high, low, close, volume  (float64 / int64)
    """
    missing = set(PRICE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"[{ticker}] Missing required columns: {missing}")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            f"[{ticker}] DataFrame index must be a DatetimeIndex, "
            f"got {type(df.index).__name__}"
        )

    df = df[PRICE_COLUMNS].copy()
    df.index.name = "date"
    df = df.sort_index()
    # Drop rows where close is NaN (happens in mixed US+EU batch downloads
    # where the shared DatetimeIndex has trading days from multiple exchanges)
    df = df.dropna(subset=["close"])
    return df

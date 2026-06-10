from __future__ import annotations

import pandas as pd
import pytest

from data.models.price_data import PRICE_COLUMNS, validate_price_df


def test_validate_happy_path(sample_price_df: pd.DataFrame) -> None:
    result = validate_price_df(sample_price_df, "AAPL")
    assert list(result.columns) == PRICE_COLUMNS
    assert isinstance(result.index, pd.DatetimeIndex)
    assert result.index.name == "date"


def test_validate_only_returns_schema_columns(sample_price_df: pd.DataFrame) -> None:
    df_extra = sample_price_df.copy()
    df_extra["extra_col"] = 999
    result = validate_price_df(df_extra, "AAPL")
    assert "extra_col" not in result.columns
    assert list(result.columns) == PRICE_COLUMNS


def test_validate_missing_column_raises(sample_price_df: pd.DataFrame) -> None:
    df = sample_price_df.drop(columns=["volume"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_price_df(df, "AAPL")


def test_validate_non_datetime_index_raises(sample_price_df: pd.DataFrame) -> None:
    df = sample_price_df.reset_index(drop=True)
    with pytest.raises(TypeError, match="DatetimeIndex"):
        validate_price_df(df, "AAPL")


def test_validate_sorts_by_date(sample_price_df: pd.DataFrame) -> None:
    df_reversed = sample_price_df.iloc[::-1]
    result = validate_price_df(df_reversed, "AAPL")
    assert result.index.is_monotonic_increasing


def test_validate_sets_index_name(sample_price_df: pd.DataFrame) -> None:
    df = sample_price_df.copy()
    df.index.name = None
    result = validate_price_df(df, "AAPL")
    assert result.index.name == "date"

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data.loaders.yfinance_loader import YFinanceLoader
from data.models.price_data import PRICE_COLUMNS


def _make_raw_df() -> pd.DataFrame:
    """Mimics the flat DataFrame that yfinance returns for a single ticker."""
    idx = pd.date_range("2024-01-02", periods=5, freq="B")
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "High": [105.0, 106.0, 107.0, 108.0, 109.0],
            "Low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "Close": [102.0, 103.0, 104.0, 105.0, 106.0],
            "Volume": [1_000_000] * 5,
        },
        index=idx,
    )


def _make_multi_raw_df(tickers: list[str]) -> pd.DataFrame:
    """
    Mimics the MultiIndex DataFrame yfinance returns for multiple tickers
    when group_by='ticker': level 0 = ticker, level 1 = price column.
    """
    raw = _make_raw_df()
    frames = {ticker: raw for ticker in tickers}
    df = pd.concat(frames, axis=1)
    df.columns.names = ["Ticker", "Price"]
    return df


@pytest.fixture
def loader() -> YFinanceLoader:
    return YFinanceLoader(timeout=10)


# ------------------------------------------------------------------
# fetch (single ticker)
# ------------------------------------------------------------------


@patch("data.loaders.yfinance_loader.yf.download")
def test_fetch_returns_canonical_schema(
    mock_dl: MagicMock, loader: YFinanceLoader
) -> None:
    mock_dl.return_value = _make_raw_df()
    df = loader.fetch("AAPL", date(2024, 1, 1), date(2024, 1, 31))
    assert list(df.columns) == PRICE_COLUMNS
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.name == "date"


@patch("data.loaders.yfinance_loader.yf.download")
def test_fetch_strips_extra_columns(mock_dl: MagicMock, loader: YFinanceLoader) -> None:
    raw = _make_raw_df()
    raw["Dividends"] = 0.0
    raw["Stock Splits"] = 0.0
    mock_dl.return_value = raw
    df = loader.fetch("AAPL", date(2024, 1, 1), date(2024, 1, 31))
    assert set(df.columns) == set(PRICE_COLUMNS)


@patch("data.loaders.yfinance_loader.yf.download")
def test_fetch_raises_on_empty_response(
    mock_dl: MagicMock, loader: YFinanceLoader
) -> None:
    mock_dl.return_value = pd.DataFrame()
    with pytest.raises(ValueError, match="No data returned"):
        loader.fetch("FAKE", date(2024, 1, 1), date(2024, 1, 31))


@patch("data.loaders.yfinance_loader.yf.download")
def test_fetch_passes_correct_params(
    mock_dl: MagicMock, loader: YFinanceLoader
) -> None:
    mock_dl.return_value = _make_raw_df()
    loader.fetch("AAPL", date(2024, 1, 1), date(2024, 1, 31))
    _, kwargs = mock_dl.call_args
    assert kwargs.get("auto_adjust") is True
    assert kwargs.get("progress") is False


# ------------------------------------------------------------------
# fetch_batch (multiple tickers)
# ------------------------------------------------------------------


@patch("data.loaders.yfinance_loader.yf.download")
def test_fetch_batch_empty_list_returns_empty(
    mock_dl: MagicMock, loader: YFinanceLoader
) -> None:
    result = loader.fetch_batch([], date(2024, 1, 1), date(2024, 1, 31))
    assert result == {}
    mock_dl.assert_not_called()


@patch("data.loaders.yfinance_loader.yf.download")
def test_fetch_batch_single_ticker(mock_dl: MagicMock, loader: YFinanceLoader) -> None:
    mock_dl.return_value = _make_raw_df()
    result = loader.fetch_batch(["AAPL"], date(2024, 1, 1), date(2024, 1, 31))
    assert "AAPL" in result
    assert list(result["AAPL"].columns) == PRICE_COLUMNS


@patch("data.loaders.yfinance_loader.yf.download")
def test_fetch_batch_multiple_tickers(
    mock_dl: MagicMock, loader: YFinanceLoader
) -> None:
    tickers = ["AAPL", "MSFT"]
    mock_dl.return_value = _make_multi_raw_df(tickers)
    result = loader.fetch_batch(tickers, date(2024, 1, 1), date(2024, 1, 31))
    for ticker in tickers:
        assert ticker in result
        assert list(result[ticker].columns) == PRICE_COLUMNS

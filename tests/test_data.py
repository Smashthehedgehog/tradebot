from datetime import date, timedelta

import pandas as pd
import pytest

from data.cache import PriceCache
from data.fetcher import fetch_prices


def test_fetch_prices_returns_dataframe():
    end = date.today().isoformat()
    start = (date.today() - timedelta(days=5)).isoformat()
    df = fetch_prices("AAPL", start, end, "1h")

    assert not df.empty
    assert "Close" in df.columns
    assert isinstance(df.index, pd.DatetimeIndex)


def test_fetch_prices_raises_on_bad_symbol():
    end = date.today().isoformat()
    start = (date.today() - timedelta(days=5)).isoformat()

    with pytest.raises(ValueError):
        fetch_prices("XXXX_INVALID", start, end, "1h")


def test_cache_put_and_get():
    cache = PriceCache()
    df = pd.DataFrame(
        {"Close": [100.0, 101.0, 102.0]},
        index=pd.date_range("2024-01-01", periods=3, freq="h"),
    )

    cache.put("AAPL", "1h", df)
    result = cache.get("AAPL", "1h")

    assert result is not None
    assert not result.empty
    pd.testing.assert_frame_equal(result, df)

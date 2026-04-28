import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def fetch_prices(symbol: str, start: str, end: str, interval: str) -> pd.DataFrame:
    """
    Download OHLCV price history for a single symbol from Yahoo Finance.

    Args:
        symbol: Ticker symbol, e.g. "AAPL".
        start: Start date string in "YYYY-MM-DD" format (inclusive).
        end: End date string in "YYYY-MM-DD" format (exclusive).
        interval: yfinance interval string, e.g. "1h" or "1d".
                  Note: "1h" is capped at 730 days of history by yfinance.

    Returns:
        DataFrame with columns [Open, High, Low, Close, Volume] and a
        DatetimeIndex. Rows where Close is NaN are dropped.

    Raises:
        ValueError: If the resulting DataFrame is empty after dropping NaNs.
    """
    raw = yf.download(
        tickers=symbol,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df = df.dropna(subset=["Close"])

    if df.empty:
        logger.warning("fetch_prices: no data returned for %s (%s to %s, %s)", symbol, start, end, interval)
        raise ValueError(f"No price data returned for {symbol} ({start} to {end}, interval={interval})")

    logger.debug("fetch_prices: %d bars fetched for %s", len(df), symbol)
    return df


def fetch_multiple(
    symbols: list[str], start: str, end: str, interval: str
) -> dict[str, pd.DataFrame]:
    """
    Download OHLCV price history for multiple symbols, returning a dict keyed by symbol.

    Symbols that fail to download are skipped with a WARNING rather than raising,
    so one bad ticker does not abort the entire fetch.

    Args:
        symbols: List of ticker symbols, e.g. ["AAPL", "MSFT"].
        start: Start date string in "YYYY-MM-DD" format (inclusive).
        end: End date string in "YYYY-MM-DD" format (exclusive).
        interval: yfinance interval string, e.g. "1h" or "1d".

    Returns:
        Dict mapping each successfully fetched symbol to its OHLCV DataFrame.
        Symbols that raised are absent from the dict.
    """
    result: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            result[symbol] = fetch_prices(symbol, start, end, interval)
        except ValueError:
            logger.warning("fetch_multiple: skipping %s — no data available", symbol)
    return result


def fetch_latest_bar(symbol: str, interval: str) -> dict:
    """
    Fetch the most recently completed price bar for a single symbol.

    Uses a 5-day lookback window to guarantee at least one completed bar is
    available regardless of weekends or holidays. If the market is currently
    open (the last bar in the response is still forming), the second-to-last
    row is returned instead of the last.

    Args:
        symbol: Ticker symbol, e.g. "AAPL".
        interval: yfinance interval string, e.g. "1h".

    Returns:
        Dict with keys: symbol, timestamp, open, high, low, close, volume.

    Raises:
        ValueError: If no completed bar can be retrieved.
    """
    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=5)

    df = fetch_prices(
        symbol,
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
        interval,
    )

    if df.empty:
        raise ValueError(f"fetch_latest_bar: no bars available for {symbol}")

    # Use the second-to-last bar when the market is open (last bar still forming).
    # A bar is considered "still forming" if its timestamp is within one interval
    # period of now. Fall back to the last bar when only one row exists.
    if len(df) >= 2 and _bar_is_incomplete(df.index[-1], interval):
        row = df.iloc[-2]
        ts = df.index[-2]
    else:
        row = df.iloc[-1]
        ts = df.index[-1]

    return {
        "symbol": symbol,
        "timestamp": ts,
        "open": float(row["Open"]),
        "high": float(row["High"]),
        "low": float(row["Low"]),
        "close": float(row["Close"]),
        "volume": float(row["Volume"]),
    }


def _bar_is_incomplete(bar_time: pd.Timestamp, interval: str) -> bool:
    """
    Return True if a bar starting at bar_time is likely still forming.

    A bar is considered incomplete if fewer than one interval period has elapsed
    since it opened. Supports "1h", "30m", "15m", "5m", "1m", and "1d".

    Args:
        bar_time: Timestamp of the bar's open.
        interval: yfinance interval string used to determine bar duration.

    Returns:
        True if the bar is still within its expected duration, else False.
    """
    interval_minutes = {
        "1m": 1, "5m": 5, "15m": 15, "30m": 30,
        "1h": 60, "1d": 1440,
    }
    duration = timedelta(minutes=interval_minutes.get(interval, 60))
    now = datetime.now(tz=timezone.utc)

    if bar_time.tzinfo is None:
        bar_time = bar_time.tz_localize("UTC")
    else:
        bar_time = bar_time.tz_convert("UTC")

    return now < bar_time + duration

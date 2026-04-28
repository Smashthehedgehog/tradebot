import logging
import os
import pickle
import threading

import pandas as pd

logger = logging.getLogger(__name__)

_CACHE_DIR = "logs"


class PriceCache:
    """
    Thread-safe in-memory cache for OHLCV price DataFrames.

    Each entry is keyed by (symbol, interval). Entries are also persisted to
    disk as pickle files so the cache survives process restarts.
    """

    def __init__(self) -> None:
        """
        Initialise an empty in-memory cache and a reentrant lock for thread safety.
        """
        self._store: dict[tuple[str, str], pd.DataFrame] = {}
        self._lock = threading.RLock()

    def put(self, symbol: str, interval: str, df: pd.DataFrame) -> None:
        """
        Store a price DataFrame in memory and persist it to disk.

        The disk file is written to logs/{symbol}_{interval}.pkl so it can be
        reloaded by load_from_disk() on the next process startup.

        Args:
            symbol: Ticker symbol, e.g. "AAPL".
            interval: yfinance interval string, e.g. "1h".
            df: OHLCV DataFrame to cache.
        """
        key = (symbol, interval)
        with self._lock:
            self._store[key] = df

        os.makedirs(_CACHE_DIR, exist_ok=True)
        path = _cache_path(symbol, interval)
        try:
            with open(path, "wb") as f:
                pickle.dump(df, f)
            logger.debug("cache.put: persisted %s/%s to %s", symbol, interval, path)
        except OSError as exc:
            logger.warning("cache.put: could not write %s — %s", path, exc)

    def get(self, symbol: str, interval: str) -> pd.DataFrame | None:
        """
        Return the cached DataFrame for a symbol/interval pair, or None if absent.

        Args:
            symbol: Ticker symbol, e.g. "AAPL".
            interval: yfinance interval string, e.g. "1h".

        Returns:
            Cached DataFrame, or None if the key has not been stored yet.
        """
        key = (symbol, interval)
        with self._lock:
            return self._store.get(key)

    def load_from_disk(self, symbol: str, interval: str) -> pd.DataFrame | None:
        """
        Attempt to load a previously pickled cache file from disk into memory.

        If the file exists and loads successfully, the DataFrame is inserted into
        the in-memory store so subsequent get() calls return it without re-reading
        disk. Returns None if the file does not exist or cannot be unpickled.

        Args:
            symbol: Ticker symbol, e.g. "AAPL".
            interval: yfinance interval string, e.g. "1h".

        Returns:
            Loaded DataFrame, or None if the file is missing or unreadable.
        """
        path = _cache_path(symbol, interval)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "rb") as f:
                df = pickle.load(f)
            with self._lock:
                self._store[(symbol, interval)] = df
            logger.debug("cache.load_from_disk: loaded %s/%s from %s", symbol, interval, path)
            return df
        except Exception as exc:
            logger.warning("cache.load_from_disk: could not read %s — %s", path, exc)
            return None

    def invalidate(self, symbol: str, interval: str) -> None:
        """
        Remove a symbol/interval entry from the in-memory store and delete its disk file.

        Does not raise if the key or file does not exist.

        Args:
            symbol: Ticker symbol, e.g. "AAPL".
            interval: yfinance interval string, e.g. "1h".
        """
        key = (symbol, interval)
        with self._lock:
            self._store.pop(key, None)

        path = _cache_path(symbol, interval)
        if os.path.exists(path):
            try:
                os.remove(path)
                logger.debug("cache.invalidate: deleted %s", path)
            except OSError as exc:
                logger.warning("cache.invalidate: could not delete %s — %s", path, exc)


def _cache_path(symbol: str, interval: str) -> str:
    """
    Return the disk path for a given symbol/interval pickle file.

    Args:
        symbol: Ticker symbol, e.g. "AAPL".
        interval: yfinance interval string, e.g. "1h".

    Returns:
        Relative file path string, e.g. "logs/AAPL_1h.pkl".
    """
    return os.path.join(_CACHE_DIR, f"{symbol}_{interval}.pkl")


# Module-level singleton — import and use this directly throughout the codebase.
price_cache = PriceCache()

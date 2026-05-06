import pandas as pd

import config
from indicators.base import BasePredictor


class RSIPredictor(BasePredictor):
    """
    Relative Strength Index: measures momentum exhaustion via average gain vs
    average loss over a rolling window. Values near 100 = overbought; near 0 = oversold.

    RSI complements MACD (which is lagging) by catching intraday reversals faster,
    and pairs with Bollinger %B to provide both price-envelope and momentum signals.
    """

    def compute(self, prices: pd.Series) -> pd.Series:
        """
        Compute RSI(RSI_WINDOW) for every bar using Wilder's smoothing.

        Uses ewm(alpha=1/RSI_WINDOW, adjust=False) to match the canonical Wilder
        smoothing definition. NaN values produced at the first bar (no prior diff)
        are forward-filled so downstream code never sees NaN.

        Args:
            prices: Close price Series with a DatetimeIndex.

        Returns:
            Series of RSI values in [0, 100], same length as prices.
        """
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        alpha = 1.0 / config.RSI_WINDOW
        avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, float("inf"))
        rsi = 100.0 - (100.0 / (1.0 + rs))
        # ffill handles any mid-series NaNs; fillna(50) covers the first bar
        # where diff() is undefined — 50 is the neutral RSI midpoint.
        return rsi.ffill().fillna(50.0)

    def signal(self, value: float) -> int:
        """
        Map an RSI value to a discrete vote.

        Thresholds: RSI < 35 → bullish (oversold momentum exhaustion, reversal likely);
                    RSI > 65 → bearish (overbought, pullback likely);
                    otherwise → neutral.

        Args:
            value: A single RSI float in [0, 100].

        Returns:
            +1, -1, or 0.
        """
        if value < 35:
            return 1
        if value > 65:
            return -1
        return 0


class SMARatioPredictor(BasePredictor):
    """
    Measures whether the current price is unusually high or low relative to
    its recent average, producing a mean-reversion signal.
    """

    def compute(self, prices: pd.Series) -> pd.Series:
        """
        Compute price / SMA(SMA_WINDOW) for every bar.

        A ratio below 1.0 means the price is below its moving average (cheap);
        above 1.0 means the price is above it (expensive). NaN values that
        arise at the start of the series (before the window fills) are
        forward-filled so no downstream code sees NaN.

        Args:
            prices: Close price Series with a DatetimeIndex.

        Returns:
            Series of price-to-SMA ratios, same length as prices.
        """
        sma = prices.rolling(window=config.SMA_WINDOW).mean()
        ratio = prices / sma
        return ratio.ffill()

    def signal(self, value: float) -> int:
        """
        Map an SMA ratio to a discrete vote.

        Thresholds: ratio < 0.95 → bullish (price well below average);
                    ratio > 1.05 → bearish (price well above average);
                    otherwise    → neutral.

        Args:
            value: A single SMA ratio float.

        Returns:
            +1, -1, or 0.
        """
        if value < 0.95:
            return 1
        if value > 1.05:
            return -1
        return 0


class BollingerPercentBPredictor(BasePredictor):
    """
    Measures where the current price sits within its normal volatility envelope,
    producing an overbought/oversold signal.
    """

    def compute(self, prices: pd.Series) -> pd.Series:
        """
        Compute Bollinger %B: (price - lower_band) / (upper_band - lower_band).

        Bands are SMA ± BBANDS_STD standard deviations over BBANDS_WINDOW bars.
        %B = 0 means price is at the lower band; %B = 1 means at the upper band.
        Values are clipped to [-0.5, 1.5] to prevent extreme distortion when
        volatility collapses and the band width approaches zero. NaN values at
        the start of the series are forward-filled.

        Args:
            prices: Close price Series with a DatetimeIndex.

        Returns:
            Series of %B values clipped to [-0.5, 1.5], same length as prices.
        """
        sma = prices.rolling(window=config.BBANDS_WINDOW).mean()
        std = prices.rolling(window=config.BBANDS_WINDOW).std()
        upper = sma + config.BBANDS_STD * std
        lower = sma - config.BBANDS_STD * std
        pct_b = (prices - lower) / (upper - lower)
        return pct_b.clip(-0.5, 1.5).ffill()

    def signal(self, value: float) -> int:
        """
        Map a %B value to a discrete vote.

        Thresholds: %B < 0.2 → bullish (price near or below lower band, oversold);
                    %B > 0.8 → bearish (price near or above upper band, overbought);
                    otherwise → neutral.

        Args:
            value: A single %B float.

        Returns:
            +1, -1, or 0.
        """
        if value < 0.2:
            return 1
        if value > 0.8:
            return -1
        return 0


class MACDHistogramPredictor(BasePredictor):
    """
    Measures short-term momentum by comparing two exponential moving averages,
    producing a trend-direction signal.
    """

    def compute(self, prices: pd.Series) -> pd.Series:
        """
        Compute the MACD histogram: (EMA_fast - EMA_slow) - EMA(MACD_line, signal).

        Uses pandas ewm() with adjust=False for all three EMAs so the first
        value is initialised from the first price rather than being NaN.
        Spans: MACD_FAST=12, MACD_SLOW=26, MACD_SIGNAL=9 (configurable).
        NaN values are forward-filled.

        Args:
            prices: Close price Series with a DatetimeIndex.

        Returns:
            Series of histogram values (fast momentum minus slow momentum),
            same length as prices.
        """
        ema_fast = prices.ewm(span=config.MACD_FAST, adjust=False).mean()
        ema_slow = prices.ewm(span=config.MACD_SLOW, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=config.MACD_SIGNAL, adjust=False).mean()
        histogram = macd_line - signal_line
        return histogram.ffill()

    def signal(self, value: float) -> int:
        """
        Map a histogram value to a discrete vote.

        A positive histogram means short-term momentum is stronger than the
        signal line (bullish); negative means the opposite (bearish).

        Args:
            value: A single MACD histogram float.

        Returns:
            +1 if value > 0, -1 if value < 0, 0 if exactly zero.
        """
        if value > 0:
            return 1
        if value < 0:
            return -1
        return 0

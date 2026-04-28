import numpy as np
import pandas as pd

import config
from indicators.technical import (
    BollingerPercentBPredictor,
    MACDHistogramPredictor,
    SMARatioPredictor,
)


def _synthetic_prices(n: int = 100, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    values = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5)
    return pd.Series(values, index=pd.date_range("2024-01-01", periods=n, freq="h"))


def test_sma_ratio_compute_shape():
    prices = _synthetic_prices(100)
    result = SMARatioPredictor().compute(prices)

    assert len(result) == len(prices)


def test_sma_ratio_signal_bounds():
    prices = _synthetic_prices(200)
    predictor = SMARatioPredictor()
    for val in predictor.compute(prices).dropna():
        assert predictor.signal(float(val)) in {-1, 0, 1}


def test_bollinger_compute_no_nan():
    prices = _synthetic_prices(100)
    result = BollingerPercentBPredictor().compute(prices)

    # After the initial warm-up period, no NaN values should remain
    after_warmup = result.iloc[config.BBANDS_WINDOW:]
    assert not after_warmup.isna().any()


def test_macd_signal_direction():
    # Monotonically increasing series — fast EMA pulls ahead of slow EMA early,
    # so the histogram should be positive during the initial divergence phase.
    prices = pd.Series(
        100.0 + np.arange(200) * 0.5,
        index=pd.date_range("2024-01-01", periods=200, freq="h"),
    )
    result = MACDHistogramPredictor().compute(prices)

    assert (result > 0).any()

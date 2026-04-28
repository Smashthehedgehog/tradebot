import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

import config
from backtest.runner import run_backtest
from engine.trading_engine import TradingEngine
from indicators.technical import (
    BollingerPercentBPredictor,
    MACDHistogramPredictor,
    SMARatioPredictor,
)
from model.predictor_manager import PredictorManager
from model.q_learner import QLearner
from model.state_encoder import StateEncoder
from model.weight_updater import WeightUpdater
from portfolio.tracker import PortfolioTracker


def _make_engine(symbols: list[str] | None = None) -> TradingEngine:
    if symbols is None:
        symbols = ["AAPL"]
    predictors = [SMARatioPredictor(), BollingerPercentBPredictor(), MACDHistogramPredictor()]
    manager = PredictorManager(predictors)
    encoder = StateEncoder(config.NUM_BINS)
    learner = QLearner(
        num_states=encoder.num_states(len(predictors)),
        num_actions=3,
        alpha=config.ALPHA,
        gamma=config.GAMMA,
        rar=config.RAR,
        radr=config.RADR,
    )
    tracker = PortfolioTracker(config.STARTING_CASH, symbols)
    updater = WeightUpdater(manager)
    return TradingEngine(symbols, manager, encoder, learner, tracker, updater)


def _fit_encoder_on_synthetic(engine: TradingEngine, n: int = 300) -> pd.Series:
    """Fit the engine's encoder on synthetic price data and return the price series."""
    rng = np.random.default_rng(0)
    prices = pd.Series(
        100.0 + np.cumsum(rng.standard_normal(n) * 0.5),
        index=pd.date_range("2023-01-01", periods=n, freq="h"),
    )
    ind_dict = {
        name: p.compute(prices)
        for name, p in engine.manager.predictors().items()
    }
    ind_df = pd.DataFrame(ind_dict).dropna()
    engine.encoder.fit(ind_df)
    return prices


def test_engine_not_trained_skips_cycle(caplog):
    engine = _make_engine()
    assert not engine.is_trained

    with caplog.at_level(logging.WARNING):
        engine.run_cycle()

    assert not engine.is_trained
    assert any("not trained" in m for m in caplog.messages)


def test_decide_returns_valid_action():
    engine = _make_engine(["AAPL"])
    prices = _fit_encoder_on_synthetic(engine)
    engine.is_trained = True

    action, shares = engine.decide("AAPL", prices)

    assert action in {"BUY", "SELL", "HOLD"}
    assert isinstance(shares, int)
    assert shares >= 0


def test_backtest_returns_metrics_dict(monkeypatch):
    monkeypatch.setattr(config, "SYMBOLS", ["AAPL"])

    engine = _make_engine(["AAPL"])
    _fit_encoder_on_synthetic(engine)
    engine.is_trained = True

    end = date.today()
    start = end - timedelta(days=30)
    result = run_backtest(engine, start.isoformat(), end.isoformat())

    assert "cumulative_return" in result
    assert "num_trades" in result
    assert "final_value" in result

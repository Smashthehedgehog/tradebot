import logging

import pytest

import config
from portfolio.tracker import PortfolioTracker

_SYMBOL = "AAPL"
_PRICE = 150.0
_SHARES = 10


def _tracker() -> PortfolioTracker:
    return PortfolioTracker(config.STARTING_CASH, [_SYMBOL])


def test_buy_reduces_cash():
    tracker = _tracker()
    tracker.execute(_SYMBOL, "BUY", _SHARES, _PRICE)

    expected_cost = _SHARES * _PRICE * (1.0 + config.IMPACT) + config.COMMISSION
    assert tracker.cash == pytest.approx(config.STARTING_CASH - expected_cost, rel=1e-9)


def test_sell_increases_cash():
    tracker = _tracker()
    tracker.execute(_SYMBOL, "BUY", _SHARES, _PRICE)
    cash_after_buy = tracker.cash

    tracker.execute(_SYMBOL, "SELL", _SHARES, _PRICE)

    # Proceeds after impact and commission added back
    proceeds = _SHARES * _PRICE * (1.0 - config.IMPACT) - config.COMMISSION
    assert tracker.cash == pytest.approx(cash_after_buy + proceeds, rel=1e-9)


def test_cannot_sell_unowned_shares(caplog):
    tracker = _tracker()
    with caplog.at_level(logging.WARNING):
        tracker.execute(_SYMBOL, "SELL", _SHARES, _PRICE)

    assert tracker.holdings[_SYMBOL] == 0
    assert tracker.cash == config.STARTING_CASH
    assert any("SELL skipped" in m for m in caplog.messages)


def test_cannot_overdraft(caplog):
    tracker = _tracker()
    prohibitively_expensive = config.STARTING_CASH * 10
    with caplog.at_level(logging.WARNING):
        tracker.execute(_SYMBOL, "BUY", _SHARES, prohibitively_expensive)

    assert tracker.cash == config.STARTING_CASH
    assert tracker.holdings[_SYMBOL] == 0
    assert any("BUY skipped" in m for m in caplog.messages)


def test_history_appended_on_trade():
    tracker = _tracker()
    assert len(tracker.history) == 0

    tracker.execute(_SYMBOL, "BUY", _SHARES, _PRICE)
    assert len(tracker.history) == 1

    tracker.execute(_SYMBOL, "SELL", _SHARES, _PRICE)
    assert len(tracker.history) == 2

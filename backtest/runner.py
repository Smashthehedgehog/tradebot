import logging

import numpy as np
import pandas as pd

import config
from data.fetcher import fetch_multiple, fetch_prices
from portfolio.tracker import PortfolioTracker

logger = logging.getLogger(__name__)


def run_backtest(engine, test_start: str, test_end: str) -> dict:
    """
    Replay the trained model over a held-out historical window and report metrics.

    Uses the same TRAIN_INTERVAL hourly bars as training to keep indicator
    values and state encoding consistent. A fresh PortfolioTracker is created
    so backtest trades are completely isolated from the live tracker's state.

    Compares model performance against a buy-and-hold of BENCHMARK_SYMBOL
    (S&P 500 via ^GSPC).

    Args:
        engine: A trained TradingEngine instance (is_trained must be True).
        test_start: Start date string "YYYY-MM-DD" (inclusive).
        test_end: End date string "YYYY-MM-DD" (exclusive).

    Returns:
        Dict with keys: cumulative_return, benchmark_cumulative_return,
        mean_daily_return, std_daily_return, num_trades, final_value.
    """
    print(f"\n[BACKTEST] Running on held-out window: {test_start} → {test_end}")

    # Fetch test-window price data for all symbols
    price_data = fetch_multiple(config.SYMBOLS, test_start, test_end, config.TRAIN_INTERVAL)
    if not price_data:
        logger.error("run_backtest: no price data fetched — aborting")
        return {}

    # Fetch benchmark data
    benchmark_return = _benchmark_return(test_start, test_end)

    # Fresh tracker — isolated from the live tracker
    bt_tracker = PortfolioTracker(config.STARTING_CASH, config.SYMBOLS)

    # Determine the shared time index across all symbols
    all_indices = [df.index for df in price_data.values()]
    shared_index = all_indices[0]
    for idx in all_indices[1:]:
        shared_index = shared_index.intersection(idx)
    shared_index = shared_index.sort_values()

    print(f"[BACKTEST] {len(shared_index)} bars across {len(price_data)} symbols\n")

    current_prices: dict[str, float] = {}

    for i, ts in enumerate(shared_index):
        for symbol, df in price_data.items():
            if ts not in df.index:
                continue

            price = float(df.loc[ts, "Close"])
            current_prices[symbol] = price

            # Build price history up to and including this bar (no lookahead)
            prices_so_far = df.loc[:ts, "Close"]
            if len(prices_so_far) < 2:
                continue

            action, shares = engine.decide(symbol, prices_so_far)
            signals = engine.manager.get_all_signals(prices_so_far)
            weighted = engine.manager.get_weighted_signal(prices_so_far)
            signals_str = " ".join(
                f"{k.replace('Predictor', '')}:{v:+d}"
                for k, v in signals.items()
            )
            reason = f"Weighted signal {weighted:+.2f} ({signals_str})"
            bt_tracker.execute(symbol, action, shares, price, reason)

    # Compute metrics
    final_value = bt_tracker.portfolio_value(current_prices)
    cumulative_return = (final_value - config.STARTING_CASH) / config.STARTING_CASH

    # Daily returns from portfolio value series
    price_series = {sym: price_data[sym]["Close"] for sym in price_data}
    daily_rets = bt_tracker.daily_returns(price_series)
    mean_daily = float(daily_rets.mean()) if not daily_rets.empty else 0.0
    std_daily = float(daily_rets.std()) if not daily_rets.empty else 0.0
    num_trades = sum(1 for h in bt_tracker.history if h["action"] != "HOLD")

    _print_metrics(
        cumulative_return=cumulative_return,
        benchmark_return=benchmark_return,
        mean_daily=mean_daily,
        std_daily=std_daily,
        num_trades=num_trades,
        final_value=final_value,
    )

    return {
        "cumulative_return": round(cumulative_return * 100, 4),
        "benchmark_cumulative_return": round(benchmark_return * 100, 4),
        "mean_daily_return": round(mean_daily * 100, 6),
        "std_daily_return": round(std_daily * 100, 6),
        "num_trades": num_trades,
        "final_value": round(final_value, 2),
    }


def _benchmark_return(test_start: str, test_end: str) -> float:
    """
    Compute the buy-and-hold cumulative return for the S&P 500 benchmark over
    the test window.

    Args:
        test_start: Start date string "YYYY-MM-DD".
        test_end: End date string "YYYY-MM-DD".

    Returns:
        Cumulative return as a decimal (e.g. 0.12 for +12%). Returns 0.0 if
        benchmark data cannot be fetched.
    """
    try:
        df = fetch_prices(config.BENCHMARK_SYMBOL, test_start, test_end, "1d")
        if df.empty or len(df) < 2:
            return 0.0
        start_price = float(df["Close"].iloc[0])
        end_price = float(df["Close"].iloc[-1])
        return (end_price - start_price) / start_price
    except Exception as exc:
        logger.warning("_benchmark_return: could not fetch benchmark — %s", exc)
        return 0.0


def _print_metrics(
    cumulative_return: float,
    benchmark_return: float,
    mean_daily: float,
    std_daily: float,
    num_trades: int,
    final_value: float,
) -> None:
    """
    Print a formatted backtest results table to stdout.

    Args:
        cumulative_return: Model cumulative return as a decimal.
        benchmark_return: S&P 500 buy-and-hold return as a decimal.
        mean_daily: Mean of daily percent returns.
        std_daily: Standard deviation of daily percent returns.
        num_trades: Total number of non-HOLD trades executed.
        final_value: Final portfolio value in USD.
    """
    print("\n" + "=" * 55)
    print("  BACKTEST RESULTS")
    print("=" * 55)
    print(f"  Final portfolio value : ${final_value:>12,.2f}")
    print(f"  Cumulative return     : {cumulative_return * 100:>+10.2f}%")
    print(f"  Benchmark (S&P 500)   : {benchmark_return * 100:>+10.2f}%")
    print(f"  Mean daily return     : {mean_daily * 100:>+10.4f}%")
    print(f"  Std of daily return   : {std_daily * 100:>10.4f}%")
    print(f"  Total trades          : {num_trades:>10}")
    print("=" * 55 + "\n")

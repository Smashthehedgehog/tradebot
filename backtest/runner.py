import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd

import config
from data.fetcher import fetch_multiple, fetch_prices
from portfolio.tracker import PortfolioTracker

logger = logging.getLogger(__name__)

# Annualisation factor: 252 trading days × 6.5 market hours per day
_HOURLY_PERIODS_PER_YEAR = 252 * 6.5


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
        mean_daily_return, std_daily_return, sharpe_ratio, max_drawdown_pct,
        num_trades, final_value.
    """
    print(f"\n[BACKTEST] Running on held-out window: {test_start} -> {test_end}")

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

    # Compute core metrics
    final_value = bt_tracker.portfolio_value(current_prices)
    cumulative_return = (final_value - config.STARTING_CASH) / config.STARTING_CASH

    price_series = {sym: price_data[sym]["Close"] for sym in price_data}
    daily_rets = bt_tracker.daily_returns(price_series)
    mean_daily = float(daily_rets.mean()) if not daily_rets.empty else 0.0
    std_daily = float(daily_rets.std()) if not daily_rets.empty else 0.0
    num_trades = sum(1 for h in bt_tracker.history if h["action"] != "HOLD")

    # Sharpe ratio (annualised at hourly cadence)
    sharpe = (
        mean_daily / std_daily * (_HOURLY_PERIODS_PER_YEAR ** 0.5)
        if std_daily > 0 else 0.0
    )

    # Maximum drawdown: largest peak-to-trough decline in the cumulative return curve
    if not daily_rets.empty:
        cumulative_curve = (1.0 + daily_rets).cumprod()
        rolling_peak = cumulative_curve.cummax()
        drawdown_series = (cumulative_curve - rolling_peak) / rolling_peak
        max_drawdown = float(drawdown_series.min())
    else:
        max_drawdown = 0.0

    _print_metrics(
        cumulative_return=cumulative_return,
        benchmark_return=benchmark_return,
        mean_daily=mean_daily,
        std_daily=std_daily,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        num_trades=num_trades,
        final_value=final_value,
    )

    alpha = cumulative_return - benchmark_return
    from notifications.emailer import send_email
    send_email(
        subject="[TradingBot] Backtest Complete",
        body=(
            f"Backtest window:  {test_start} -> {test_end}\n"
            f"Symbols:          {', '.join(config.SYMBOLS)}\n\n"
            f"Final value:      ${final_value:,.2f}\n"
            f"Cumulative:       {cumulative_return * 100:+.2f}%\n"
            f"Benchmark S&P500: {benchmark_return * 100:+.2f}%\n"
            f"Alpha:            {alpha * 100:+.2f}%\n\n"
            f"Mean daily:       {mean_daily * 100:+.4f}%\n"
            f"Std daily:        {std_daily * 100:.4f}%\n"
            f"Sharpe ratio:     {sharpe:.3f}\n"
            f"Max drawdown:     {max_drawdown * 100:.2f}%\n"
            f"Total trades:     {num_trades}"
        ),
    )

    return {
        "cumulative_return": round(cumulative_return * 100, 4),
        "benchmark_cumulative_return": round(benchmark_return * 100, 4),
        "mean_daily_return": round(mean_daily * 100, 6),
        "std_daily_return": round(std_daily * 100, 6),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown_pct": round(max_drawdown * 100, 4),
        "num_trades": num_trades,
        "final_value": round(final_value, 2),
    }


def run_walk_forward(
    engine,
    full_start: str,
    full_end: str,
    test_window_days: int = 45,
    n_folds: int = 4,
) -> list[dict]:
    """
    Run anchored walk-forward validation across n_folds independent test windows.

    For each fold the training window expands from full_start to the fold's test
    start. The test window is the next test_window_days days. This gives multiple
    independent performance samples and a far more reliable estimate of whether
    the model generalises than a single train/test split.

    Args:
        engine: TradingEngine instance (will be fully retrained for each fold).
        full_start: Start of all available data "YYYY-MM-DD".
        full_end: End of all available data "YYYY-MM-DD" (last test window ends here).
        test_window_days: Calendar days in each test window (default 45).
        n_folds: Number of folds to run (default 4).

    Returns:
        List of metric dicts, one per fold, each matching run_backtest() output
        plus a "fold" key with the 1-based fold index.
    """
    results: list[dict] = []
    end = date.fromisoformat(full_end)

    for fold in range(n_folds, 0, -1):
        test_end = end - timedelta(days=(fold - 1) * test_window_days)
        test_start = test_end - timedelta(days=test_window_days)
        train_end = test_start
        train_start = date.fromisoformat(full_start)

        if train_start >= train_end:
            logger.warning(
                "run_walk_forward: fold %d skipped — train window too small",
                n_folds - fold + 1,
            )
            continue

        fold_num = n_folds - fold + 1
        print(
            f"\n[WF] Fold {fold_num}/{n_folds}"
            f" | Train: {train_start} -> {train_end}"
            f" | Test:  {test_start} -> {test_end}"
        )

        engine.retrain(train_start.isoformat(), train_end.isoformat())
        metrics = run_backtest(engine, test_start.isoformat(), test_end.isoformat())
        metrics["fold"] = fold_num
        results.append(metrics)

    if not results:
        logger.error("run_walk_forward: no folds completed")
        return results

    avg_return = sum(r["cumulative_return"] for r in results) / len(results)
    avg_benchmark = sum(r["benchmark_cumulative_return"] for r in results) / len(results)
    avg_sharpe = sum(r.get("sharpe_ratio", 0.0) for r in results) / len(results)

    print("\n" + "=" * 55)
    print("  WALK-FORWARD SUMMARY")
    print("=" * 55)
    print(f"  Folds completed        : {len(results)}/{n_folds}")
    print(f"  Avg cumulative return  : {avg_return:>+10.2f}%")
    print(f"  Avg benchmark return   : {avg_benchmark:>+10.2f}%")
    print(f"  Avg Sharpe ratio       : {avg_sharpe:>10.3f}")
    print("=" * 55 + "\n")

    fold_lines = "\n".join(
        f"  Fold {r['fold']}: return {r['cumulative_return']:+.2f}%"
        f" | benchmark {r['benchmark_cumulative_return']:+.2f}%"
        f" | alpha {r['cumulative_return'] - r['benchmark_cumulative_return']:+.2f}%"
        f" | Sharpe {r.get('sharpe_ratio', 0.0):.3f}"
        f" | trades {r.get('num_trades', 0)}"
        for r in results
    )
    from notifications.emailer import send_email
    send_email(
        subject="[TradingBot] Walk-Forward Validation Complete",
        body=(
            f"Walk-forward: {n_folds} folds, {test_window_days}-day test windows\n\n"
            f"Per-fold results:\n{fold_lines}\n\n"
            f"Averages:\n"
            f"  Return:    {avg_return:+.2f}%\n"
            f"  Benchmark: {avg_benchmark:+.2f}%\n"
            f"  Alpha:     {avg_return - avg_benchmark:+.2f}%\n"
            f"  Sharpe:    {avg_sharpe:.3f}"
        ),
    )

    return results


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
    sharpe: float,
    max_drawdown: float,
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
        sharpe: Annualised Sharpe ratio.
        max_drawdown: Maximum peak-to-trough drawdown as a decimal (negative).
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
    print(f"  Sharpe ratio (ann.)   : {sharpe:>10.3f}")
    print(f"  Max drawdown          : {max_drawdown * 100:>+10.2f}%")
    print(f"  Total trades          : {num_trades:>10}")
    print("=" * 55 + "\n")

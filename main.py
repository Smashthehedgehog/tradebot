import argparse
import logging
import logging.handlers
import os
import sys
from datetime import date, timedelta

import uvicorn

import config
from api.server import app, init as api_init
from backtest.runner import run_backtest, run_walk_forward
from engine.scheduler import TradingScheduler
from engine.trading_engine import TradingEngine
from indicators.technical import (
    BollingerPercentBPredictor,
    MACDHistogramPredictor,
    RSIPredictor,
    SMARatioPredictor,
)
from model.predictor_manager import PredictorManager
from model.q_learner import QLearner
from model.state_encoder import StateEncoder
from model.weight_updater import WeightUpdater
from portfolio.tracker import PortfolioTracker


class _Tee:
    """Mirrors every write to sys.stdout into a log file simultaneously."""

    def __init__(self, terminal, log_file):
        self._terminal = terminal
        self._log_file = log_file

    def write(self, data: str) -> None:
        self._terminal.write(data)
        self._log_file.write(data)

    def flush(self) -> None:
        self._terminal.flush()
        self._log_file.flush()

    def fileno(self) -> int:
        return self._terminal.fileno()


def _configure_logging() -> None:
    """
    Configure the root logger to write to both stdout and a rotating log file.

    All output — both logger records and bare print() calls — is also mirrored
    to logs/console.log via a _Tee so the full console history is preserved on
    disk for later inspection.

    Log records are emitted in a structured format that includes timestamp,
    level, logger name, and message. The file handler rotates at 10 MB and
    keeps 5 backups so the logs/ directory does not grow unbounded.
    """
    os.makedirs("logs", exist_ok=True)

    console_log = open("logs/console.log", "a", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, console_log)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )

    file_handler = logging.handlers.RotatingFileHandler(
        "logs/trading_bot.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)


def _parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments that control startup behaviour.

    Returns:
        Namespace with attributes: retrain (bool), backtest_only (bool),
        train_start (str | None), train_end (str | None).
    """
    parser = argparse.ArgumentParser(
        description="ML Trading Bot — tabular Q-learning on yfinance hourly data"
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        default=False,
        help="Force a full cold-start retrain even if a saved Q-table exists.",
    )
    parser.add_argument(
        "--backtest-only",
        action="store_true",
        default=False,
        help="Train (or load), run the backtest, print metrics, then exit.",
    )
    parser.add_argument(
        "--train-start",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help=(
            f"Override training start date. Must be within {config.TRAIN_LOOKBACK_DAYS} "
            "days of today to stay within yfinance's hourly data limit."
        ),
    )
    parser.add_argument(
        "--train-end",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="Override training end date (defaults to today minus BACKTEST_DAYS).",
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        default=False,
        help=(
            "Replace the single backtest with 4-fold anchored walk-forward validation. "
            "Implies --backtest-only: exits after printing fold results."
        ),
    )
    return parser.parse_args()


def _compute_date_windows(args: argparse.Namespace) -> tuple[str, str, str, str]:
    """
    Compute training and backtest date windows, applying CLI overrides where provided.

    Default windows (always relative to today):
      Training  : today − TRAIN_LOOKBACK_DAYS  ->  today − BACKTEST_DAYS
      Backtest  : today − BACKTEST_DAYS         ->  today

    Warns if the resolved train_start is more than TRAIN_LOOKBACK_DAYS ago, since
    yfinance caps hourly history at 730 days.

    Args:
        args: Parsed CLI arguments (train_start, train_end overrides).

    Returns:
        Tuple of (train_start, train_end, backtest_start, backtest_end) as
        "YYYY-MM-DD" strings.
    """
    logger = logging.getLogger(__name__)
    today = date.today()

    backtest_end = today
    backtest_start = today - timedelta(days=config.BACKTEST_DAYS)
    train_end_default = backtest_start
    train_start_default = today - timedelta(days=config.TRAIN_LOOKBACK_DAYS)

    train_start = args.train_start if args.train_start else train_start_default.isoformat()
    train_end = args.train_end if args.train_end else train_end_default.isoformat()

    # Warn if requested start exceeds yfinance hourly data limit
    cutoff = today - timedelta(days=config.TRAIN_LOOKBACK_DAYS)
    if date.fromisoformat(train_start) < cutoff:
        logger.warning(
            "_compute_date_windows: train_start %s is older than %d days — "
            "yfinance may not return hourly bars that far back",
            train_start,
            config.TRAIN_LOOKBACK_DAYS,
        )

    return train_start, train_end, backtest_start.isoformat(), backtest_end.isoformat()


def main() -> None:
    """
    Entry point for the trading bot backend.

    Startup sequence:
      1. Parse CLI flags.
      2. Configure logging.
      3. Instantiate all components (manager, encoder, learner, tracker, updater, engine).
      4. Compute dynamic training and backtest date windows.
      5. Cold or warm start: retrain if --retrain passed or no saved Q-table found.
      6. Run the backtest and print metrics.
      7. If --backtest-only, exit here.
      8. Wire the FastAPI app with component references.
      9. Start the APScheduler background scheduler.
      10. Start the uvicorn server (blocks until shutdown).
    """
    args = _parse_args()
    _configure_logging()
    logger = logging.getLogger(__name__)

    # Install email handler for ERROR-level log records (no-ops if email not configured)
    from notifications.emailer import EmailErrorHandler
    email_handler = EmailErrorHandler(level=logging.ERROR)
    email_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logging.getLogger().addHandler(email_handler)

    logger.info("main: trading bot starting up")

    # ------------------------------------------------------------------
    # Step 3 — Instantiate components
    # ------------------------------------------------------------------
    predictors = [
        SMARatioPredictor(),
        BollingerPercentBPredictor(),
        MACDHistogramPredictor(),
        RSIPredictor(),
    ]
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
    tracker = PortfolioTracker(config.STARTING_CASH, config.SYMBOLS)
    updater = WeightUpdater(manager)
    engine = TradingEngine(
        symbols=config.SYMBOLS,
        manager=manager,
        encoder=encoder,
        learner=learner,
        tracker=tracker,
        updater=updater,
    )

    # ------------------------------------------------------------------
    # Step 4 — Compute date windows
    # ------------------------------------------------------------------
    train_start, train_end, backtest_start, backtest_end = _compute_date_windows(args)
    logger.info(
        "main: training window %s -> %s | backtest window %s -> %s",
        train_start,
        train_end,
        backtest_start,
        backtest_end,
    )

    # ------------------------------------------------------------------
    # Step 5 — Cold / warm start
    # ------------------------------------------------------------------
    if args.retrain:
        logger.info("main: --retrain flag set — forcing cold start")
        engine.retrain(train_start, train_end)
    else:
        loaded = engine.load_pretrained()
        if not loaded:
            logger.info("main: no saved Q-table — running cold start training")
            engine.train(train_start, train_end)
        else:
            print("[STARTUP] Warm start — skipping training")

    # ------------------------------------------------------------------
    # Step 6 — Backtest (single-fold or walk-forward)
    # ------------------------------------------------------------------
    if not engine.is_trained:
        logger.error("main: training did not complete — skipping backtest and exiting")
        return

    if args.walk_forward:
        logger.info("main: --walk-forward flag set — running 4-fold walk-forward validation")
        run_walk_forward(engine, train_start, backtest_end)
        logger.info("main: walk-forward complete — exiting")
        return

    run_backtest(engine, backtest_start, backtest_end)

    # ------------------------------------------------------------------
    # Step 7 — Exit if --backtest-only
    # ------------------------------------------------------------------
    if args.backtest_only:
        logger.info("main: --backtest-only flag set — exiting after backtest")
        return

    # ------------------------------------------------------------------
    # Steps 8–10 — Wire API, start scheduler, start uvicorn
    # ------------------------------------------------------------------
    scheduler = TradingScheduler(engine)
    api_init(engine, tracker, manager, updater, scheduler)

    scheduler.start()
    logger.info("main: scheduler started")

    print("[STARTUP] FastAPI server starting on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")


if __name__ == "__main__":
    main()

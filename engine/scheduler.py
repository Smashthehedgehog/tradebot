import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from apscheduler.schedulers.background import BackgroundScheduler

import config

logger = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")


class TradingScheduler:
    """
    Wraps APScheduler to fire engine.run_cycle() on a fixed interval, but only
    during market hours on weekdays.
    """

    def __init__(self, engine) -> None:
        """
        Initialise the scheduler with a reference to the trading engine.

        Args:
            engine: TradingEngine instance whose run_cycle() will be called.
        """
        self._engine = engine
        self._scheduler = BackgroundScheduler(timezone="America/New_York")

    def start(self) -> None:
        """
        Register the interval job and start the background scheduler.

        The job fires every SCHEDULE_INTERVAL_MINUTES minutes. Each firing
        calls is_market_hours() first and silently returns if the market is
        closed, so the scheduler can run continuously without any external
        cron management.
        """
        self._scheduler.add_job(
            self._conditional_cycle,
            trigger="interval",
            minutes=config.SCHEDULE_INTERVAL_MINUTES,
            id="trading_cycle",
        )
        self._scheduler.start()
        logger.info(
            "scheduler.start: firing every %d min during market hours",
            config.SCHEDULE_INTERVAL_MINUTES,
        )

    def stop(self) -> None:
        """
        Shut down the background scheduler gracefully without waiting for
        running jobs to complete.
        """
        self._scheduler.shutdown(wait=False)
        logger.info("scheduler.stop: scheduler shut down")

    def is_market_hours(self) -> bool:
        """
        Return True if the current US/Eastern time falls within market hours
        on a weekday (Monday–Friday, MARKET_OPEN_HOUR to MARKET_CLOSE_HOUR).

        Does not account for market holidays; those will result in a cycle that
        fetches data and finds no new bars, which is handled gracefully by the
        engine's fetch_latest_bar logic.

        Returns:
            True if the market is currently open, False otherwise.
        """
        now = datetime.now(tz=_ET)
        if now.weekday() >= 5:  # Saturday=5, Sunday=6
            return False
        return config.MARKET_OPEN_HOUR <= now.hour < config.MARKET_CLOSE_HOUR

    def _conditional_cycle(self) -> None:
        """
        Internal job function: run the engine cycle only during market hours.
        """
        if not self.is_market_hours():
            logger.debug("scheduler: outside market hours — skipping cycle")
            return
        try:
            self._engine.run_cycle()
        except Exception as exc:
            logger.error("scheduler: unhandled error in run_cycle — %s", exc, exc_info=True)

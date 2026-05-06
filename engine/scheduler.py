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
    during actual market hours on trading days (excluding US market holidays).
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
        closed or it is a holiday, so the scheduler can run continuously
        without any external cron management.
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
        Return True if the NYSE is currently open, accounting for weekends,
        market holidays, and early-close days.

        Uses pandas_market_calendars to query the official NYSE schedule.
        Falls back to the simple weekday + hour check if the library raises,
        so a missing dependency never silently prevents live trading.

        Returns:
            True if the market is currently open, False otherwise.
        """
        now = datetime.now(tz=_ET)
        try:
            import pandas_market_calendars as mcal
            nyse = mcal.get_calendar("NYSE")
            today_str = now.strftime("%Y-%m-%d")
            schedule = nyse.schedule(start_date=today_str, end_date=today_str)
            if schedule.empty:
                return False  # weekend or holiday
            import pytz
            et = pytz.timezone("US/Eastern")
            market_open = schedule.iloc[0]["market_open"].tz_convert(et)
            market_close = schedule.iloc[0]["market_close"].tz_convert(et)
            # Convert now to pytz-aware for comparison
            now_pytz = datetime.now(tz=et)
            return market_open <= now_pytz <= market_close
        except ImportError:
            logger.warning(
                "scheduler: pandas_market_calendars not installed — "
                "falling back to simple weekday/hour check (no holiday awareness)"
            )
            if now.weekday() >= 5:
                return False
            return config.MARKET_OPEN_HOUR <= now.hour < config.MARKET_CLOSE_HOUR
        except Exception as exc:
            logger.warning(
                "scheduler: calendar lookup failed (%s) — falling back to simple check", exc
            )
            if now.weekday() >= 5:
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

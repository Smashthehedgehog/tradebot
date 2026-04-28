import logging
import threading
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Module-level singletons — set by init() before uvicorn starts
_engine = None
_tracker = None
_manager = None
_updater = None
_scheduler = None


def init(engine, tracker, manager, updater, scheduler) -> None:
    """
    Wire the module-level singletons before the server starts accepting requests.

    Must be called from main.py after all components are constructed but before
    uvicorn.run() is invoked. Routes access these globals directly so no
    dependency-injection framework is needed.

    Args:
        engine: Trained TradingEngine instance.
        tracker: Live PortfolioTracker instance.
        manager: PredictorManager holding all active indicators.
        updater: WeightUpdater for EMA accuracy tracking.
        scheduler: TradingScheduler instance (used for is_market_hours()).
    """
    global _engine, _tracker, _manager, _updater, _scheduler
    _engine = engine
    _tracker = tracker
    _manager = manager
    _updater = updater
    _scheduler = scheduler


app = FastAPI(title="Trading Bot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------
# Request models
# ------------------------------------------------------------------


class RetrainRequest(BaseModel):
    train_start: str
    train_end: str


class SetWeightRequest(BaseModel):
    predictor: str
    weight: float


# ------------------------------------------------------------------
# GET routes
# ------------------------------------------------------------------


@app.get("/status")
def get_status() -> dict:
    """
    Return high-level engine state.

    Returns:
        Dict with keys:
          - trained (bool): Whether the Q-table has been fitted.
          - is_market_hours (bool): Whether the market is currently open.
          - timestamp (str): Current UTC time in ISO format.
    """
    is_market = _scheduler.is_market_hours() if _scheduler is not None else False
    return {
        "trained": _engine.is_trained if _engine is not None else False,
        "is_market_hours": is_market,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }


@app.get("/holdings")
def get_holdings() -> dict:
    """
    Return current portfolio state from the live tracker.

    Uses the engine's last recorded prices for valuation. Falls back to an
    empty price map (cash-only valuation) if no prices have been seen yet.

    Returns:
        Dict from tracker.summary() containing total_value, cash, holdings,
        pnl, pnl_pct, and num_trades.
    """
    if _tracker is None:
        raise HTTPException(status_code=503, detail="Tracker not initialised")
    current_prices = _engine._prev_prices if _engine is not None else {}
    return _tracker.summary(current_prices)


@app.get("/weights")
def get_weights() -> dict:
    """
    Return current predictor weights and their EMA accuracy scores.

    Returns:
        Dict with keys:
          - weights (dict[str, float]): Current normalised weight per predictor.
          - ema_accuracies (dict[str, float]): Exponential moving average of
            each predictor's recent correctness (range 0.0–1.0).
    """
    if _manager is None:
        raise HTTPException(status_code=503, detail="Manager not initialised")
    return {
        "weights": _manager.get_weights(),
        "ema_accuracies": _updater.get_ema_accuracies() if _updater is not None else {},
    }


@app.get("/history")
def get_history() -> list:
    """
    Return the complete trade history from the live tracker.

    Returns:
        List of trade record dicts, each containing symbol, action, shares,
        price, reason, and timestamp fields.
    """
    if _tracker is None:
        raise HTTPException(status_code=503, detail="Tracker not initialised")
    return _tracker.history


@app.get("/history/{symbol}")
def get_history_for_symbol(symbol: str) -> list:
    """
    Return trade history filtered to a single ticker symbol.

    Args:
        symbol: Ticker symbol to filter by (e.g. "AAPL"). Case-sensitive.

    Returns:
        Subset of tracker.history where the symbol field matches. Returns an
        empty list if the symbol has no recorded trades.
    """
    if _tracker is None:
        raise HTTPException(status_code=503, detail="Tracker not initialised")
    return [h for h in _tracker.history if h.get("symbol") == symbol]


# ------------------------------------------------------------------
# POST routes
# ------------------------------------------------------------------


@app.post("/retrain")
def post_retrain(body: RetrainRequest) -> dict:
    """
    Trigger a full cold-start retrain in a background thread.

    The retrain runs asynchronously so this route returns immediately.
    Monitor stdout for [RETRAIN] and [TRAIN] progress lines.

    Args:
        body: JSON body with train_start ("YYYY-MM-DD") and train_end ("YYYY-MM-DD").

    Returns:
        Dict with key status set to "retraining_started".
    """
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialised")

    def _retrain():
        try:
            _engine.retrain(body.train_start, body.train_end)
        except Exception as exc:
            logger.error("post_retrain: retrain thread failed — %s", exc, exc_info=True)

    thread = threading.Thread(target=_retrain, daemon=True, name="retrain-thread")
    thread.start()
    logger.info("post_retrain: retrain thread started for %s → %s", body.train_start, body.train_end)
    return {"status": "retraining_started"}


@app.post("/set_weight")
def post_set_weight(body: SetWeightRequest) -> dict:
    """
    Override one predictor's weight, then re-normalise across all predictors.

    The updated weights sum to 1.0 after normalisation. Any weight below
    MIN_PREDICTOR_WEIGHT is clamped before normalisation.

    Args:
        body: JSON body with predictor (str) and weight (float).

    Returns:
        Dict with key weights containing the full updated weight mapping.
    """
    if _manager is None:
        raise HTTPException(status_code=503, detail="Manager not initialised")
    _manager.set_weight(body.predictor, body.weight)
    logger.info("post_set_weight: set '%s' weight to %.4f", body.predictor, body.weight)
    return {"weights": _manager.get_weights()}

# Trading Model Implementation Specification

## Purpose

This document is a complete, step-by-step implementation specification for an
autonomous machine learning trading model. Feed this document to an AI coding
assistant to generate the full codebase. Every step must be implemented in order.
No account connectivity or frontend integration is included — all trade decisions
are printed to stdout. The system is designed to run as a backend Python server
that a React frontend can later connect to via a REST API.

---

## Guiding Principles

- **Modularity first.** Every predictor, indicator, and model component lives in
  its own file and can be swapped or extended without touching other files.
- **Self-tuning.** The system tracks which predictors are right and wrong over time
  and updates their weights automatically.
- **Testable end-to-end.** Every layer (data fetch, indicators, model, engine) has
  unit tests. A full backtest can be run from the command line with a single command.
- **No live brokerage.** All trade decisions are printed with full detail — symbol,
  action (BUY / SELL / HOLD), quantity, price, rationale. Nothing is executed.
- **Hourly cadence.** A scheduler fires the decision loop every hour during market
  hours.
- **Documented.** Every function must have a docstring describing parameters,
  return values, and the reason the function exists.

---

## Technology Stack

| Layer | Library |
|---|---|
| Market data | `yfinance` |
| Numerical computation | `numpy`, `pandas` |
| ML / RL model | Custom tabular Q-learner (pure numpy) |
| Online weight updates | Custom exponential moving average tracker |
| API server | `FastAPI` + `uvicorn` |
| Scheduling | `APScheduler` |
| Testing | `pytest` |
| Logging | Python `logging` (structured JSON output) |

---

## Project File Structure

Generate every file listed below. Do not add extra files beyond this list.

```
trading_bot/
├── main.py                         # Entry point — starts server + scheduler
├── config.py                       # All tunable constants in one place
├── requirements.txt                # Pinned dependencies
│
├── data/
│   ├── __init__.py
│   ├── fetcher.py                  # yfinance data fetching
│   └── cache.py                    # In-memory + disk cache for price data
│
├── indicators/
│   ├── __init__.py
│   ├── base.py                     # Abstract base class for all predictors
│   └── technical.py                # SMA ratio, Bollinger %B, MACD histogram
│
├── model/
│   ├── __init__.py
│   ├── state_encoder.py            # Discretize continuous indicator values → state int
│   ├── q_learner.py                # Tabular Q-learning engine
│   ├── predictor_manager.py        # Registry of active predictors + their weights
│   └── weight_updater.py           # Online algorithm that adjusts predictor weights
│
├── portfolio/
│   ├── __init__.py
│   └── tracker.py                  # Simulated holdings, cash, P&L — no real trades
│
├── engine/
│   ├── __init__.py
│   ├── trading_engine.py           # Orchestrates one decision cycle
│   └── scheduler.py                # APScheduler wrapper, fires engine hourly
│
├── backtest/
│   ├── __init__.py
│   └── runner.py                   # Runs a full historical backtest, prints results
│
├── api/
│   ├── __init__.py
│   └── server.py                   # FastAPI routes (status, holdings, weights, history)
│
├── tests/
│   ├── test_data.py
│   ├── test_indicators.py
│   ├── test_model.py
│   ├── test_portfolio.py
│   └── test_engine.py
│
└── logs/
    └── .gitkeep
```

---

## Step 1 — `requirements.txt`

Generate a `requirements.txt` with pinned versions for:

```
yfinance>=0.2.40
pandas>=2.0.0
numpy>=1.26.0
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
APScheduler>=3.10.4
pytest>=8.0.0
httpx>=0.27.0          # for FastAPI test client
```

---

## Step 2 — `config.py`

This file is the single source of truth for every tunable value. No magic numbers
anywhere else in the codebase — always import from config.

### Variables to define

```python
# --- Symbols ---
SYMBOLS: list[str]         # e.g. ["AAPL", "MSFT", "GOOG"] — stocks the model trades
BENCHMARK_SYMBOL: str      # e.g. "^GSPC" — S&P 500

# --- Capital ---
STARTING_CASH: float       # e.g. 100_000.0
MAX_POSITION_SHARES: int   # e.g. 100 — max shares per position per symbol
TRADE_UNIT: int            # e.g. 10 — shares per order

# --- Market friction (simulated) ---
COMMISSION: float          # e.g. 0.0 — set to 0 until live trading
IMPACT: float              # e.g. 0.001 — simulated slippage

# --- Q-Learner hyperparameters ---
NUM_BINS: int              # e.g. 8 — quantile bins per indicator
ALPHA: float               # e.g. 0.2 — Q-learning rate
GAMMA: float               # e.g. 0.9 — discount factor
RAR: float                 # e.g. 0.5 — initial random action rate
RADR: float                # e.g. 0.9999 — random action decay per step
MAX_EPOCHS: int            # e.g. 50 — training epochs
EARLY_STOP_PATIENCE: int   # e.g. 5 — epochs without improvement before stopping
EARLY_STOP_DELTA: float    # e.g. 1e-4 — minimum reward improvement threshold

# --- Indicators ---
SMA_WINDOW: int            # e.g. 20
BBANDS_WINDOW: int         # e.g. 20
BBANDS_STD: float          # e.g. 2.0
MACD_FAST: int             # e.g. 12
MACD_SLOW: int             # e.g. 26
MACD_SIGNAL: int           # e.g. 9

# --- Weight updater ---
WEIGHT_DECAY: float        # e.g. 0.95 — exponential decay for historical accuracy
MIN_PREDICTOR_WEIGHT: float # e.g. 0.05 — floor so no predictor is zeroed out

# --- Scheduler ---
MARKET_OPEN_HOUR: int      # e.g. 9  (9 AM ET)
MARKET_CLOSE_HOUR: int     # e.g. 16 (4 PM ET)
SCHEDULE_INTERVAL_MINUTES: int # e.g. 60

# --- Training ---
# Training uses hourly bars to keep granularity consistent with live decision cycles.
# yfinance caps hourly data at 730 days back — TRAIN_START must stay within that limit.
# TRAIN_START and TRAIN_END should be computed dynamically at startup (see main.py Step 16)
# as: TRAIN_END = today, TRAIN_START = today minus 730 days.
# The last BACKTEST_DAYS of that window are held out as a test set; the rest is training.
TRAIN_LOOKBACK_DAYS: int   # e.g. 730 — total days of hourly history to fetch (yfinance max for "1h")
BACKTEST_DAYS: int         # e.g. 90  — days held out from the end of the window for backtesting
                           # Training window = TRAIN_LOOKBACK_DAYS − BACKTEST_DAYS days of hourly bars
                           # Backtest window = final BACKTEST_DAYS days of hourly bars
                           # Both windows are computed dynamically from today's date at startup.
TRAIN_INTERVAL: str        # e.g. "1h" — yfinance interval used for both training and live cycles
                           # Using the same interval for training and live data keeps indicator
                           # values and bin edges consistent — no scale mismatch between training
                           # and inference. yfinance limit: max 730 days back for "1h".
LIVE_INTERVAL: str         # e.g. "1h" — yfinance interval used by the live decision cycle
                           # Must match TRAIN_INTERVAL so the Q-table state space is consistent.
QTABLE_PATH: str           # e.g. "logs/qtable.pkl" — path to save/load the trained Q-table
```

Every variable must have an inline comment explaining its purpose and valid range.

---

## Step 3 — `data/fetcher.py`

### `fetch_prices(symbol, start, end, interval) -> pd.DataFrame`

Fetches OHLCV data from yfinance for a single symbol.

- Parameters: `symbol (str)`, `start (str)`, `end (str)`, `interval (str)`
- Returns: DataFrame with columns `[Open, High, Low, Close, Volume]`, DatetimeIndex
- Drops rows where Close is NaN
- Raises `ValueError` if the resulting DataFrame is empty
- Log a DEBUG message on success, a WARNING on empty result before raising

### `fetch_multiple(symbols, start, end, interval) -> dict[str, pd.DataFrame]`

Calls `fetch_prices` for each symbol and returns a dict keyed by symbol.

- Silently skips any symbol that raises and logs a WARNING for it

### `fetch_latest_bar(symbol, interval) -> dict`

Fetches the most recent completed bar for a symbol.

- Returns a dict with keys: `symbol`, `timestamp`, `open`, `high`, `low`, `close`, `volume`
- Uses a lookback of 5 days to guarantee at least one completed bar is returned
- Extracts the second-to-last row if the market is currently open (last bar incomplete)
- Otherwise extracts the last row

---

## Step 4 — `data/cache.py`

### `class PriceCache`

Thread-safe in-memory cache for price DataFrames.

#### Methods

- `put(symbol, interval, df)` — Store a DataFrame under a `(symbol, interval)` key. Also pickle it to `logs/{symbol}_{interval}.pkl` for persistence across restarts.
- `get(symbol, interval) -> pd.DataFrame | None` — Return cached DataFrame or None if absent.
- `load_from_disk(symbol, interval) -> pd.DataFrame | None` — Try to load a previously pickled cache file. Return None if file does not exist.
- `invalidate(symbol, interval)` — Remove a key from the in-memory dict and delete the disk file if present.

There must be a module-level singleton: `price_cache = PriceCache()`.

---

## Step 5 — `indicators/base.py`

### `class BasePredictor` (Abstract)

Every predictor must subclass this.

#### Abstract methods

- `compute(prices: pd.Series) -> pd.Series` — Given a Close price series, return a same-length Series of continuous signal values. Must not introduce lookahead (only use data up to each timestamp).
- `signal(value: float) -> int` — Map a single continuous value to a trade vote: `+1` (bullish), `-1` (bearish), or `0` (neutral).

#### Concrete methods

- `name(self) -> str` — Returns the class name. Used as the predictor's registry key.
- `votes(prices: pd.Series) -> pd.Series` — Calls `compute` then maps each value through `signal`. Returns an integer Series.

#### Attributes

- `weight: float` — Current weight of this predictor in range [0, 1]. Default 1.0.
- `accuracy_history: list[float]` — Rolling list of per-step accuracy values (1.0 if the vote matched the future return direction, 0.0 otherwise). Maximum 500 entries (FIFO).

---

## Step 6 — `indicators/technical.py`

Implement three concrete predictors, each subclassing `BasePredictor`.

### `class SMARatioPredictor`

Computes `price / SMA(SMA_WINDOW)`.

- `compute`: Rolling mean over `SMA_WINDOW` days. Divide price by SMA. Forward-fill NaN.
- `signal`: Returns `+1` if ratio < 0.95 (price below SMA, potential mean-reversion long), `-1` if ratio > 1.05, else `0`.

### `class BollingerPercentBPredictor`

Computes Bollinger %B: `(price - lower_band) / (upper_band - lower_band)`.

- `compute`: SMA ± `BBANDS_STD` standard deviations over `BBANDS_WINDOW`. %B clipped to [-0.5, 1.5] to avoid division-by-zero distortion. Forward-fill NaN.
- `signal`: Returns `+1` if %B < 0.2, `-1` if %B > 0.8, else `0`.

### `class MACDHistogramPredictor`

Computes MACD histogram: `(EMA_fast - EMA_slow) - EMA(MACD_line, MACD_SIGNAL)`.

- `compute`: Use `pd.Series.ewm(span=N, adjust=False).mean()` for all EMAs. Forward-fill NaN.
- `signal`: Returns `+1` if histogram > 0, `-1` if histogram < 0, else `0`.

All three classes must have full docstrings on `compute` and `signal` explaining
the formula and the thresholds.

---

## Step 7 — `model/predictor_manager.py`

### `class PredictorManager`

Central registry that holds all active predictor instances and their weights.

#### Constructor `__init__(predictors: list[BasePredictor])`

- Stores predictors in an `OrderedDict` keyed by `predictor.name()`
- Each predictor starts with `weight = 1.0 / len(predictors)` (equal weighting)

#### Methods

- `get_weighted_signal(prices: pd.Series) -> float` — For the latest timestamp, compute each predictor's vote, multiply by its weight, and return the weighted sum. Range: [-1.0, +1.0].
- `get_all_signals(prices: pd.Series) -> dict[str, int]` — Return a dict of `{predictor_name: vote}` for the latest bar. Used for logging.
- `get_weights() -> dict[str, float]` — Return a dict of `{predictor_name: weight}`. Used by the API.
- `set_weight(name: str, weight: float)` — Manually override one predictor's weight. Re-normalizes all weights so they sum to 1.0. Clamps to `MIN_PREDICTOR_WEIGHT`.
- `register(predictor: BasePredictor)` — Add a new predictor at runtime. Redistributes weights equally. This is the extension point for adding news or other predictors.
- `record_accuracy(prices: pd.Series, realized_return: float)` — Append a 1.0 or 0.0 to each predictor's `accuracy_history` based on whether its vote matched the sign of `realized_return`.

---

## Step 8 — `model/weight_updater.py`

### `class WeightUpdater`

Automatically adjusts predictor weights based on recent accuracy.

#### Constructor `__init__(manager: PredictorManager)`

- Holds a reference to the `PredictorManager`
- Maintains a `step_count: int` counter

#### Methods

- `update()` — Called after every completed trading step. For each predictor:
  1. Compute its exponential moving average accuracy using `WEIGHT_DECAY`:
     `ema_acc = WEIGHT_DECAY * previous_ema + (1 - WEIGHT_DECAY) * latest_accuracy`
  2. Set the raw weight proportional to `ema_acc`
  3. After updating all raw weights, normalize them (divide by sum) and clamp each to `MIN_PREDICTOR_WEIGHT`
  4. Write updated weights back via `manager.set_weight`
  5. Log the new weights at DEBUG level
- `get_ema_accuracies() -> dict[str, float]` — Return the current EMA accuracy for each predictor. Used by the API.

---

## Step 9 — `model/state_encoder.py`

### `class StateEncoder`

Converts continuous indicator values into a discrete Q-table state index.

#### Constructor `__init__(num_bins: int)`

- `num_bins` is loaded from config
- `bin_edges: list[np.ndarray] | None` — Initialized to None; populated during `fit`

#### Methods

- `fit(indicator_df: pd.DataFrame)` — Given a DataFrame where each column is a continuous indicator time series, compute `NUM_BINS` equal-frequency (quantile) bin edges per column using `np.quantile`. Store edges in `self.bin_edges`. Must be called on training data only. Raise `RuntimeError` if called more than once without explicit reset.
- `reset()` — Clear `bin_edges` so `fit` can be called again (useful for retraining).
- `restore(bin_edges: list[np.ndarray])` — Directly set `self.bin_edges` from a previously
  saved value. Used by warm-start loading so the encoder is consistent with the saved Q-table
  without needing to re-fetch training data.
- `encode(indicator_values: dict[str, float], holding: int) -> int` — Given current indicator values (one float per predictor name) and a holding index (0=flat, 1=long, 2=short), compute the bin index for each indicator, combine them into a single state int, and add the holding offset. Returns a single integer in `[0, num_states)`.
  - Formula: `state = holding * (NUM_BINS ** num_indicators) + bin_combo`
  - Clamp all bin indices to `[0, NUM_BINS - 1]`
- `num_states(num_indicators: int) -> int` — Returns `3 * (NUM_BINS ** num_indicators)`. Pure utility used when constructing the Q-table.

---

## Step 10 — `model/q_learner.py`

### `class QLearner`

Tabular Q-learning engine. Pure numpy. No external ML dependencies.

#### Constructor `__init__(num_states, num_actions, alpha, gamma, rar, radr)`

- Initialize Q-table as `np.zeros((num_states, num_actions))` with small random noise: `Q += np.random.uniform(-0.001, 0.001, Q.shape)`
- Store all hyperparameters as instance attributes
- `self.s: int` — Current state (initialized to 0)
- `self.a: int` — Last action taken (initialized to 0)

#### Methods

- `querysetstate(s: int) -> int` — Set `self.s = s`. Select action using ε-greedy (random with prob `rar`, else `argmax Q[s]`). Store as `self.a`. Return `self.a`. Does NOT update Q-table. Call this for the first step of each epoch.
- `query(s_prime: int, r: float) -> int` — Update Q-table for the previous `(self.s, self.a)` transition:
  `Q[s, a] += alpha * (r + gamma * max(Q[s_prime]) - Q[s, a])`. Then decay `rar *= radr`. Set `self.s = s_prime`, select and store next action, return it.
- `best_action(s: int) -> int` — Return `argmax Q[s]` with NO randomness and NO Q-update. Used during inference only.
- `save(path: str, bin_edges: list[np.ndarray] | None = None)` — Pickle a dict containing
  the Q-table AND `bin_edges` to disk: `{"qtable": self.Q, "bin_edges": bin_edges}`. Storing
  bin edges alongside the Q-table ensures warm-start loading is always self-consistent.
- `load(path: str) -> list[np.ndarray] | None` — Load the pickle from disk. Restore
  `self.Q` from the saved Q-table. Return the saved `bin_edges` (or None if the file was
  saved without them). Raise `FileNotFoundError` with a helpful message if the file is missing.
- `reset()` — Re-initialize the Q-table and reset rar to its original value. Allows retraining.

---

## Step 11 — `portfolio/tracker.py`

### `class PortfolioTracker`

Simulates cash and holdings. No real orders are placed.

#### Constructor `__init__(starting_cash: float, symbols: list[str])`

- `self.cash: float` — Starts at `starting_cash`
- `self.holdings: dict[str, int]` — `{symbol: shares_held}`, initialized to 0 for each symbol
- `self.history: list[dict]` — Append-only log of every action taken
- `self.starting_cash: float` — Stored for performance calculations

#### Methods

- `execute(symbol: str, action: str, shares: int, price: float)` — Simulate a trade:
  - `action` is one of `"BUY"`, `"SELL"`, `"HOLD"`
  - For BUY: deduct `shares * price * (1 + IMPACT) + COMMISSION` from cash. Add shares to holdings.
  - For SELL: add `shares * price * (1 - IMPACT) - COMMISSION` to cash. Subtract shares from holdings.
  - For HOLD: no-op on cash or holdings.
  - Validate: do not allow cash to go negative. Do not allow selling more shares than held. Log a WARNING and skip if either condition is violated.
  - Append a record to `self.history` with: `timestamp`, `symbol`, `action`, `shares`, `price`, `cash_after`, `holdings_snapshot`
  - **Print to stdout** a formatted summary of every action, including the reason it was taken. Format: `[TRADE] {timestamp} | {symbol} | {action} | {shares} shares @ ${price:.2f} | Cash: ${cash:.2f}`
- `portfolio_value(current_prices: dict[str, float]) -> float` — Return `cash + sum(shares * price for each holding)`.
- `daily_returns(price_history: dict[str, pd.Series]) -> pd.Series` — Reconstruct portfolio value time series from `history` and return its daily percent change. Used for performance metrics.
- `summary(current_prices: dict[str, float]) -> dict` — Return a dict containing: `total_value`, `cash`, `holdings`, `pnl`, `pnl_pct`, `num_trades`. Used by the API.

---

## Step 12 — `engine/trading_engine.py`

### `class TradingEngine`

Orchestrates one full decision cycle. Called once per hour during market hours.

#### Constructor `__init__(symbols, manager, encoder, learner, tracker)`

- Accepts all major components as constructor arguments (dependency injection)
- `self.is_trained: bool` — False until `train()` completes

#### Methods

- `train(train_start, train_end, force=False)` — Full historical pre-training loop using
  hourly bars (`TRAIN_INTERVAL="1h"`). This is the cold-start path.
  1. **Print** `[TRAIN] Starting historical pre-training: {train_start} → {train_end}` to stdout.
  2. Fetch hourly OHLCV data for all symbols via `fetcher.fetch_multiple(symbols, train_start, train_end, TRAIN_INTERVAL)`.
     - Hourly bars are limited to 730 days back by yfinance. `train_start` is always computed
       dynamically as `today − TRAIN_LOOKBACK_DAYS` days to stay within this limit.
     - Print a progress line per symbol: `[TRAIN] Fetched {N} hourly bars for {symbol}`.
  3. For each symbol, compute all indicator time series (SMA ratio, Bollinger %B, MACD histogram)
     over the full training window using `manager`'s predictors.
  4. Call `encoder.fit()` on the concatenated indicator DataFrame (fit on training data only —
     never on test or live data, to prevent lookahead bias).
  5. Run the Q-learning epoch loop (up to `MAX_EPOCHS`):
     - At the start of each epoch, reset position to flat and call `learner.querysetstate(initial_state)`.
     - Walk forward bar by bar through the training window for each symbol.
     - At each bar: encode current state → get action → compute reward → call `learner.query`.
     - Reward formula: `position * pct_return_next_bar - IMPACT * abs(position_change)`
     - Accumulate total reward for the epoch.
     - **Print** epoch progress: `[TRAIN] Epoch {e}/{MAX_EPOCHS} | Reward: {reward:.4f} | rar: {rar:.4f}`
     - Apply early stopping: if reward improvement < `EARLY_STOP_DELTA` for
       `EARLY_STOP_PATIENCE` consecutive epochs, break and print
       `[TRAIN] Early stop at epoch {e} — reward stable`.
  6. Save the Q-table to `QTABLE_PATH` via `learner.save(QTABLE_PATH)`.
  7. Set `self.is_trained = True`.
  8. **Print** `[TRAIN] Training complete. Epochs: {e} | Final reward: {reward:.4f} | Elapsed: {t:.1f}s`.
  9. After all epochs, call `weight_updater.update()` once using accuracy accumulated during training.
- `decide(symbol: str, prices: pd.Series) -> tuple[str, int]` — For a single symbol, given recent price history:
  1. Compute current indicator values (latest bar only)
  2. Get current holding from tracker
  3. Encode state
  4. Call `learner.best_action(state)` — deterministic, no randomness
  5. Map action int (0=HOLD, 1=BUY, 2=SELL) to `(action_str, shares)`
  6. Return `(action_str, shares)`
- `run_cycle()` — One full hourly decision cycle using live bars (`LIVE_INTERVAL`):
  1. Check `self.is_trained`; if not, log a WARNING and return immediately.
  2. For each symbol in `SYMBOLS`:
     a. Fetch the most recent completed bar via `fetcher.fetch_latest_bar(symbol, LIVE_INTERVAL)`.
     b. Retrieve recent price history from `price_cache`. If cache is empty for this symbol,
        fetch the last 60 days of `LIVE_INTERVAL` bars to populate it.
     c. Compute realized return since the last cycle bar (for accuracy tracking).
     d. Call `manager.record_accuracy(prices, realized_return)`.
     e. Call `weight_updater.update()`.
     f. Call `decide(symbol, prices)` — uses `encoder` fitted on daily training data;
        the continuous indicator values are scale-invariant so daily-trained bins
        apply correctly to hourly bars.
     g. Call `tracker.execute(symbol, action, shares, current_price)` — prints to stdout.
  3. Log cycle completion: `[CYCLE] {timestamp} | Portfolio: ${value:.2f}`.
- `load_pretrained() -> bool` — Attempt to load an existing Q-table from `QTABLE_PATH` via
  `learner.load(QTABLE_PATH)`. If the file exists and loads successfully, set
  `self.is_trained = True`, print `[TRAIN] Loaded pre-trained Q-table from {QTABLE_PATH}`,
  and return `True`. If the file does not exist, print
  `[TRAIN] No saved Q-table found at {QTABLE_PATH} — cold start required` and return `False`.
- `retrain(train_start, train_end)` — Call `encoder.reset()`, `learner.reset()`, delete the
  existing Q-table file if present, then call `train(train_start, train_end)`. Allows
  full retraining on a new or extended date range. Print
  `[RETRAIN] Retraining from scratch on {train_start} → {train_end}` before starting.

---

## Step 13 — `engine/scheduler.py`

### `class TradingScheduler`

Wraps APScheduler to fire `engine.run_cycle()` every hour during market hours.

#### Constructor `__init__(engine: TradingEngine)`

- Creates a `BackgroundScheduler` from APScheduler
- Uses US/Eastern timezone

#### Methods

- `start()` — Add an interval job: every `SCHEDULE_INTERVAL_MINUTES` minutes. The job function calls `engine.run_cycle()` only if current time is between `MARKET_OPEN_HOUR` and `MARKET_CLOSE_HOUR` (Monday–Friday). Start the scheduler.
- `stop()` — Shut down the scheduler gracefully.
- `is_market_hours() -> bool` — Returns True if current ET time is within market hours on a weekday. Pure utility, no side effects.

---

## Step 14 — `backtest/runner.py`

### `run_backtest(engine, test_start, test_end, interval) -> dict`

A standalone function that runs the trained model over a historical test window
without touching the live scheduler or tracker.

1. Fetch price data for `test_start` to `test_end` for all symbols
2. Create a fresh `PortfolioTracker` with `STARTING_CASH`
3. Walk forward bar by bar through the test window:
   - For each bar, call `engine.decide(symbol, prices_up_to_this_bar)`
   - Call the fresh tracker's `execute()` to simulate the trade (prints to stdout)
4. At the end, compute and **print** the following metrics:
   - Cumulative return (model vs benchmark)
   - Mean daily return
   - Std of daily return
   - Total number of trades
   - Final portfolio value
5. Return all metrics as a dict

The benchmark is a buy-and-hold of `BENCHMARK_SYMBOL` (S&P 500 via `^GSPC`).

---

## Step 15 — `api/server.py`

### FastAPI server

Create a FastAPI app with the following GET routes. All routes return JSON.
The engine and tracker are module-level singletons passed in at startup.

- `GET /status` — Returns `{"trained": bool, "is_market_hours": bool, "timestamp": str}`
- `GET /holdings` — Returns current holdings and cash from `tracker.summary()`
- `GET /weights` — Returns predictor weights from `manager.get_weights()` and EMA accuracies from `weight_updater.get_ema_accuracies()`
- `GET /history` — Returns the full `tracker.history` list (all past actions)
- `GET /history/{symbol}` — Returns history filtered to a single symbol
- `POST /retrain` — Accepts JSON body `{"train_start": str, "train_end": str}`. Triggers `engine.retrain()` in a background thread. Returns `{"status": "retraining_started"}`
- `POST /set_weight` — Accepts JSON body `{"predictor": str, "weight": float}`. Calls `manager.set_weight()`. Returns the updated weight dict.

Enable CORS for all origins (React dev server will call this from localhost).

---

## Step 16 — `main.py`

Entry point. This is what you run on the backend server.

### CLI flags

Parse command-line arguments using `argparse` before doing anything else:

- `--retrain` (flag, default False) — Force a full cold-start retrain even if a saved
  Q-table already exists on disk. Without this flag, a saved Q-table is loaded and
  training is skipped (warm start).
- `--backtest-only` (flag, default False) — Skip the scheduler and API server. Train (or
  load), run the backtest, print metrics, and exit. Useful for validating model quality
  without starting the live system.
- `--train-start` (str, default: today minus `TRAIN_LOOKBACK_DAYS`) — Override the training start date. Must be within 730 days of today to stay within yfinance's hourly data limit.
- `--train-end` (str, default: today minus `BACKTEST_DAYS`) — Override the training end date.

### Startup sequence

1. Configure root logger to write structured JSON to `logs/trading_bot.log` and to stdout.
2. Instantiate all components in order:
   ```
   predictors = [SMARatioPredictor(), BollingerPercentBPredictor(), MACDHistogramPredictor()]
   manager    = PredictorManager(predictors)
   encoder    = StateEncoder(NUM_BINS)
   learner    = QLearner(encoder.num_states(len(predictors)), num_actions=3, ...)
   tracker    = PortfolioTracker(STARTING_CASH, SYMBOLS)
   updater    = WeightUpdater(manager)
   engine     = TradingEngine(SYMBOLS, manager, encoder, learner, tracker, updater)
   ```
3. **Compute training and backtest date windows dynamically** (always based on today):
   ```
   today        = date.today()
   train_start  = today - timedelta(days=TRAIN_LOOKBACK_DAYS)   # e.g. 730 days ago
   backtest_end = today
   backtest_start = today - timedelta(days=BACKTEST_DAYS)        # e.g. 90 days ago
   train_end    = backtest_start                                  # training stops where backtest begins
   ```
   If `--train-start` or `--train-end` CLI overrides are provided, use those instead.
   Warn if the requested start date is more than 730 days ago (yfinance hourly limit).

4. **Cold/warm start decision:**
   - If `--retrain` was passed OR `engine.load_pretrained()` returns `False`:
     - Run `engine.train(train_start, train_end)` — fetches up to 730 days of hourly
       yfinance data and runs the full Q-learning epoch loop. Prints progress to stdout.
   - Otherwise:
     - Q-table was loaded from disk. Print `[STARTUP] Warm start — skipping training`.
     - The encoder bin edges must also be refit on training data so state encoding is
       consistent with what the Q-table learned. If no saved bin edges exist alongside
       the Q-table, force a retrain (log an ERROR and retrain automatically).
       To avoid this, `learner.save()` must also serialize and save `encoder.bin_edges`
       into the same pickle file. `learner.load()` must restore them and call
       `encoder.restore(bin_edges)` — add `restore(bin_edges)` to `StateEncoder`.
5. Run a backtest on the held-out window and print results:
   `run_backtest(engine, backtest_start, backtest_end)` — uses the same `TRAIN_INTERVAL="1h"`
   hourly bars, consistent with how the model was trained.
6. If `--backtest-only` was passed, exit here.
7. Start the scheduler: `TradingScheduler(engine).start()`
8. Start the FastAPI server: `uvicorn.run(app, host="0.0.0.0", port=8000)`

---

## Step 17 — Tests

### `tests/test_data.py`

- `test_fetch_prices_returns_dataframe` — Fetch 5 days of AAPL 1h data. Assert DataFrame is not empty, has a Close column, and index is DatetimeIndex.
- `test_fetch_prices_raises_on_bad_symbol` — Assert `ValueError` is raised for symbol `"XXXX_INVALID"`.
- `test_cache_put_and_get` — Put a small DataFrame, get it back, assert equality.

### `tests/test_indicators.py`

- `test_sma_ratio_compute_shape` — Compute on 100-bar synthetic price series. Assert output length equals input.
- `test_sma_ratio_signal_bounds` — Assert signal returns only -1, 0, or +1.
- `test_bollinger_compute_no_nan` — After forward-fill, assert no NaN in result.
- `test_macd_signal_direction` — On a monotonically increasing price series, assert MACD histogram is eventually positive.

### `tests/test_model.py`

- `test_state_encoder_fit_encode` — Fit on synthetic indicator data, encode one row, assert result is int in valid range.
- `test_qlearner_query_updates_qtable` — After `query()`, assert Q-table value for the updated cell changed.
- `test_qlearner_best_action_deterministic` — Call `best_action` twice with same state, assert same result.
- `test_predictor_manager_weights_sum_to_one` — After construction and after `set_weight`, assert sum of weights ≈ 1.0.
- `test_weight_updater_changes_weights` — Record several accuracy events, call `update()`, assert weights changed.

### `tests/test_portfolio.py`

- `test_buy_reduces_cash` — Execute a BUY, assert cash decreased.
- `test_sell_increases_cash` — Execute a BUY then SELL, assert cash is near starting value (minus friction).
- `test_cannot_sell_unowned_shares` — Assert SELL of unowned shares is ignored (no crash, WARNING logged).
- `test_cannot_overdraft` — Assert BUY that exceeds cash is ignored.
- `test_history_appended_on_trade` — Assert `len(history)` increases by 1 after each non-HOLD trade.

### `tests/test_engine.py`

- `test_engine_not_trained_skips_cycle` — Assert `run_cycle()` logs a WARNING and returns without error when `is_trained=False`.
- `test_backtest_returns_metrics_dict` — Run a short 30-day backtest on AAPL, assert returned dict has keys: `cumulative_return`, `num_trades`, `final_value`.
- `test_decide_returns_valid_action` — After training, call `decide()`, assert returned action is one of `"BUY"`, `"SELL"`, `"HOLD"`.

---

## How to Run

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run tests
```bash
pytest tests/ -v
```

### First-time cold start — fetch historical yfinance data and train
```bash
python main.py --retrain
```
Downloads years of daily price bars from yfinance (e.g. 2015–2023), runs the
Q-learning training loop, saves the Q-table to `logs/qtable.pkl`, runs a backtest
on the configured test window, then starts the server and scheduler.

### Subsequent warm starts — skip training, load saved Q-table
```bash
python main.py
```
Loads `logs/qtable.pkl` and the saved bin edges, runs the backtest, then starts
the server and scheduler. Training is skipped entirely.

### Validate model quality without starting the server
```bash
python main.py --backtest-only
```
Loads (or trains if no Q-table exists), runs the backtest, prints all metrics
and simulated trades to stdout, then exits. No scheduler or server is started.

### Retrain on a different date range
```bash
python main.py --retrain --train-start 2010-01-01 --train-end 2022-12-31
```

The server starts at `http://localhost:8000`. All trade decisions print to stdout.
API docs are auto-generated at `http://localhost:8000/docs`.

---

## How to Add a New Predictor

This is the primary extension point. Example: adding a news sentiment predictor.

1. Create a new class in `indicators/technical.py` (or a new file) that subclasses `BasePredictor`.
2. Implement `compute(prices)` — return a pd.Series of continuous sentiment scores.
3. Implement `signal(value)` — map the sentiment score to +1, 0, or -1.
4. In `main.py`, append an instance of your new class to the `predictors` list.
5. Retrain the model (`engine.retrain(...)`) to incorporate the new state dimensions.

No other files need to change.

---

## How to Retrain

Call the `/retrain` API endpoint:

```bash
curl -X POST http://localhost:8000/retrain \
  -H "Content-Type: application/json" \
  -d '{"train_start": "2022-01-01", "train_end": "2024-12-31"}'
```

Or call `engine.retrain(train_start, train_end, interval)` directly in code.

---

## Documentation Standard

Every function must have a docstring in the following format:

```python
def function_name(param1: type, param2: type) -> return_type:
    """
    One-sentence description of what this function does and why it exists.

    Args:
        param1: Description and valid range/values.
        param2: Description and valid range/values.

    Returns:
        Description of the return value and its structure.

    Raises:
        ValueError: When and why this is raised.
    """
```

No function body should be longer than 60 lines. Split into helpers if needed.

---

## Output Contract (Printed to Stdout)

Every trade decision must print in exactly this format so the frontend can later
parse it:

```
[TRADE] 2024-06-15 14:00:00 ET | AAPL | BUY | 10 shares @ $213.45 | Cash: $97,865.50 | Reason: Weighted signal +0.72 (SMA:+1 BB:+1 MACD:0)
[TRADE] 2024-06-15 14:00:00 ET | MSFT | HOLD | 0 shares @ $425.10 | Cash: $97,865.50 | Reason: Weighted signal +0.05 (SMA:0 BB:+1 MACD:-1)
```

HOLD actions must also be printed. Every line must be parseable by splitting on `|`.

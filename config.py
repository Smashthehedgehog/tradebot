# config.py
# Single source of truth for all tunable constants.
# No magic numbers anywhere else in the codebase — always import from here.

# --- Symbols ---
SYMBOLS: list[str] = ["AAPL", "MSFT", "GOOG"]  # Stocks the model trades
BENCHMARK_SYMBOL: str = "^GSPC"                 # S&P 500 — used as the performance benchmark

# --- Capital ---
STARTING_CASH: float = 100_000.0   # Simulated starting cash in USD
MAX_POSITION_SHARES: int = 100      # Maximum shares held per symbol at any time
TRADE_UNIT: int = 10                # Number of shares per buy or sell order

# --- Market friction (simulated) ---
COMMISSION: float = 0.0    # Fixed cost per trade in USD; set to 0 until live trading
IMPACT: float = 0.001      # Simulated slippage as a fraction of price (0.1% per side)

# --- Q-Learner hyperparameters ---
NUM_BINS: int = 8               # Number of equal-frequency quantile bins per indicator;
                                # total states = 3 holdings × 8³ indicators = 1,536
ALPHA: float = 0.2              # Learning rate: how much each new experience updates the Q-table;
                                # range (0, 1] — higher = faster but noisier learning
GAMMA: float = 0.9              # Discount factor: how much future rewards are valued vs immediate;
                                # range [0, 1] — 0.9 balances short and long term
RAR: float = 0.5                # Initial random action rate: probability of exploring randomly
                                # at the start of training; range [0, 1]
RADR: float = 0.9999            # Random action decay rate: RAR multiplied by this each step;
                                # at 0.9999 over ~50 epochs the model shifts from explore to exploit
MAX_EPOCHS: int = 50            # Maximum number of full passes through training data
EARLY_STOP_PATIENCE: int = 5    # Stop training if reward does not improve for this many epochs
EARLY_STOP_DELTA: float = 1e-4  # Minimum reward improvement required to reset patience counter

# --- Indicators ---
SMA_WINDOW: int = 20        # Rolling window (in bars) for the Simple Moving Average ratio signal
BBANDS_WINDOW: int = 20     # Rolling window (in bars) for Bollinger Band calculation
BBANDS_STD: float = 2.0     # Number of standard deviations above/below SMA for the bands
MACD_FAST: int = 12         # Fast EMA span (in bars) for MACD calculation
MACD_SLOW: int = 26         # Slow EMA span (in bars) for MACD calculation
MACD_SIGNAL: int = 9        # Signal EMA span (in bars) applied to the MACD line

# --- Weight updater ---
WEIGHT_DECAY: float = 0.95       # Exponential decay applied to accuracy history;
                                  # closer to 1.0 = slower adaptation, closer to 0 = faster
MIN_PREDICTOR_WEIGHT: float = 0.05  # Floor weight for any predictor; prevents any signal
                                     # from being completely silenced; range (0, 1/n_predictors)

# --- Scheduler ---
MARKET_OPEN_HOUR: int = 9       # Market open hour in US/Eastern time (9 AM)
MARKET_CLOSE_HOUR: int = 16     # Market close hour in US/Eastern time (4 PM)
SCHEDULE_INTERVAL_MINUTES: int = 60  # How often the decision cycle fires, in minutes

# --- Training ---
# Training uses hourly bars ("1h") to keep granularity consistent with live decision cycles.
# yfinance caps hourly data at 730 days back — TRAIN_LOOKBACK_DAYS must not exceed this.
# Date windows are computed dynamically at startup based on today's date (see main.py).
#   Training window : today − TRAIN_LOOKBACK_DAYS  →  today − BACKTEST_DAYS
#   Backtest window : today − BACKTEST_DAYS         →  today
TRAIN_LOOKBACK_DAYS: int = 730   # Total calendar days of hourly history to fetch;
                                  # 730 is the maximum yfinance allows for the "1h" interval
BACKTEST_DAYS: int = 90          # Days held out from the end of the window for the practice test;
                                  # these bars are never seen during training
TRAIN_INTERVAL: str = "1h"       # yfinance interval for both training and live cycles;
                                  # must stay "1h" to remain within the 730-day yfinance limit
LIVE_INTERVAL: str = "1h"        # yfinance interval used by the live hourly decision cycle;
                                  # must match TRAIN_INTERVAL so state encoding is consistent
QTABLE_PATH: str = "logs/qtable.pkl"  # File path where the trained Q-table and bin edges
                                       # are saved after training and loaded on warm start

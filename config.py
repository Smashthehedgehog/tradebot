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

# --- Trade mode ---
# Long-only keeps training and live inference consistent: action 2 exits to flat,
# never opens a short. Change to "long_short" only if the brokerage supports shorting.
TRADE_MODE: str = "long_only"

# --- Market friction (simulated) ---
COMMISSION: float = 0.0    # Fixed cost per trade in USD; set to 0 until live trading
IMPACT: float = 0.001      # Simulated slippage as a fraction of price (0.1% per side)

# --- Q-Learner hyperparameters ---
NUM_BINS: int = 6               # Number of equal-frequency quantile bins per indicator;
                                # with 4 indicators: 3 holdings × 6^4 = 3,888 total states
ALPHA: float = 0.2              # Learning rate: how much each new experience updates the Q-table;
                                # range (0, 1] — higher = faster but noisier learning
GAMMA: float = 0.9              # Discount factor: how much future rewards are valued vs immediate;
                                # range [0, 1] — 0.9 balances short and long term
RAR: float = 0.5                # Initial random action rate: probability of exploring randomly
                                # at the start of training; range [0, 1]
RADR: float = 0.9999            # Random action decay rate: RAR multiplied by this each step;
                                # at 0.9999 over ~50 epochs the model shifts from explore to exploit
MAX_EPOCHS: int = 50            # Maximum number of full passes through training data
EARLY_STOP_PATIENCE: int = 8    # Stop training if reward does not improve for this many epochs;
                                # raised from 5 to reduce premature stopping during noisy exploration
EARLY_STOP_DELTA: float = 5e-5  # Minimum reward improvement required to reset patience counter;
                                # lowered from 1e-4 to avoid false plateau detection

# --- Indicators ---
# All windows are in hourly bars. 6.5 market hours per day → 1 week ≈ 32–33 bars.
SMA_WINDOW: int = 40        # Rolling window for SMA ratio signal; ~1 week of hourly bars
BBANDS_WINDOW: int = 40     # Rolling window for Bollinger Band calculation; ~1 week of hourly bars
BBANDS_STD: float = 2.0     # Number of standard deviations above/below SMA for the bands
# MACD parameters scaled for hourly data to approximate daily 12/26/9 behaviour:
MACD_FAST: int = 24         # Fast EMA span; ~3 trading days at hourly cadence
MACD_SLOW: int = 52         # Slow EMA span; ~6.5 trading days at hourly cadence
MACD_SIGNAL: int = 18       # Signal EMA span; ~2.25 trading days at hourly cadence
RSI_WINDOW: int = 14        # RSI lookback period; 14 hourly bars ≈ 2 trading days

# --- Weight updater ---
WEIGHT_DECAY: float = 0.97       # Exponential decay applied to accuracy history;
                                  # half-life ~23 steps ≈ ~3 trading days at hourly cadence;
                                  # raised from 0.95 to stabilise weights against single-bar noise
MIN_PREDICTOR_WEIGHT: float = 0.05  # Floor weight for any predictor; prevents any signal
                                     # from being completely silenced; range (0, 1/n_predictors)

# --- Scheduler ---
MARKET_OPEN_HOUR: int = 10      # First cycle at 10 AM ET — market opens at 9:30 AM, so the
                                 # 9:30–10:30 bar is complete and available from yfinance by 10 AM
MARKET_CLOSE_HOUR: int = 16     # Market close hour in US/Eastern time (4 PM)
SCHEDULE_INTERVAL_MINUTES: int = 60  # How often the decision cycle fires, in minutes

# --- Training ---
# Training uses hourly bars ("1h") to keep granularity consistent with live decision cycles.
# yfinance caps hourly data at 730 days back — TRAIN_LOOKBACK_DAYS must not exceed this.
# Date windows are computed dynamically at startup based on today's date (see main.py).
#   Training window : today − TRAIN_LOOKBACK_DAYS  →  today − BACKTEST_DAYS
#   Backtest window : today − BACKTEST_DAYS         →  today
TRAIN_LOOKBACK_DAYS: int = 600   # Total calendar days of hourly history to fetch;
                                  # yfinance allows up to 730 days for the "1h" interval
BACKTEST_DAYS: int = 90          # Days held out from the end of the window for the practice test;
                                  # these bars are never seen during training
TRAIN_INTERVAL: str = "1h"       # yfinance interval for both training and live cycles;
                                  # must stay "1h" to remain within the 730-day yfinance limit
LIVE_INTERVAL: str = "1h"        # yfinance interval used by the live hourly decision cycle;
                                  # must match TRAIN_INTERVAL so state encoding is consistent
QTABLE_PATH: str = "logs/qtable.pkl"  # File path where the trained Q-table and bin edges
                                       # are saved after training and loaded on warm start

# --- Email notifications ---
# Leave SMTP_USER or NOTIFY_EMAIL blank to disable email entirely (safe default).
# For Gmail: enable 2-factor auth, generate an App Password, use smtp.gmail.com:587.
# For SendGrid: use smtp.sendgrid.net:587, SMTP_USER="apikey", SMTP_PASS=<api_key>.
SMTP_HOST: str = ""          # SMTP server hostname, e.g. "smtp.gmail.com"
SMTP_PORT: int = 587         # Standard STARTTLS port
SMTP_USER: str = ""          # Sender email address / SMTP login username
SMTP_PASS: str = ""          # SMTP password or app-specific password
NOTIFY_EMAIL: str = ""       # Recipient address for all bot notifications

# Trading Bot — Improvement Specification

This document details every concrete improvement to make to the existing trading model
for better hourly decision-making, stronger backtest reliability, and production-readiness.
Feed this document to an AI coding assistant to implement the changes in order. Each item
includes the exact problem, why it matters for hourly trading, and the precise fix.

---

## How to Read This Document

Each section follows the same structure:

- **Problem** — what is wrong or missing right now
- **Why it matters** — the concrete impact on decision quality or live accuracy
- **Fix** — exactly what to change and how

Items are ordered by priority: critical correctness fixes first, then quality improvements,
then production features.

---

## Part 1 — Critical Correctness Fixes

These are bugs or mismatches that silently degrade model quality. Fix these before anything else.

---

### Fix 1 — Resolve the Training / Live Position Asymmetry

**Problem**

During training in `engine/trading_engine.py` (`_run_epoch`), the model learns three positions:
`+1` (long), `-1` (short), and `0` (flat). Action `2` sets `new_position = -1` (go short).
During live inference in `decide()`, action `2` maps to `"SELL"` which only liquidates existing
longs — it never opens a short. The model trains on reward signals from short positions it is
physically unable to execute at runtime. This means roughly one-third of the Q-table's learned
behavior is never callable live, and the training rewards are systematically misleading.

**Why it matters for hourly trading**

Hourly trading depends heavily on the model knowing when NOT to hold. If short positions are
rewarded during training but mapped to no-ops live, the model is not learning what it is actually
executing. The Q-values for the short-position states are trained on fictitious outcomes. The model
will appear to learn, but its live behavior will not match what training reinforced.

**Fix**

Choose one of two approaches and implement it consistently across both `_run_epoch` and `decide()`:

**Option A — Long-only mode (simpler, recommended for now)**

In `_run_epoch`, remove the short position. Map action `2` to flat (exit long) rather than short.
Update the reward formula accordingly. In `decide()`, map action `2` to "SELL all held shares"
(same as current). In `config.py`, add a comment: `TRADE_MODE = "long_only"`. This makes training
and live behavior identical.

```python
# In _run_epoch, change the action mapping:
if action == 1:
    new_position = 1   # go long
elif action == 2:
    new_position = 0   # go flat (exit)
else:
    new_position = position  # hold
```

**Option B — Enable real short-selling live**

Add `short_positions: dict[str, int]` to `PortfolioTracker`. In `decide()`, when `action_int == 2`
and `held == 0`, open a short by borrowing shares. This requires margin accounting in the tracker.
Only choose this option if you plan to connect to a brokerage that supports shorting.

For now, implement Option A.

---

### Fix 2 — Correct Market Open Hour

**Problem**

`config.py` sets `MARKET_OPEN_HOUR = 9`. The US stock market opens at 9:30 AM Eastern, not
9:00 AM. The scheduler in `engine/scheduler.py` fires the cycle whenever the current hour is
between 9 and 16. This means the 9 AM cycle fires 30 minutes before the market opens. The
`fetch_latest_bar` call at 9 AM will return the previous session's last bar (from Friday if
today is Monday), and the model will make a decision based on stale data right before open.

**Why it matters**

For hourly trading the first cycle of the day is critical — it often captures the opening gap.
Using pre-market stale data for that first decision teaches the model nothing useful and burns
a trading opportunity.

**Fix**

In `config.py`, change `MARKET_OPEN_HOUR = 9` to `MARKET_OPEN_HOUR = 10`. This shifts the
first live cycle to 10 AM Eastern, guaranteeing the market has been open for 30 minutes and
the first hourly bar (9:30–10:30) is complete and available from yfinance. Update the README
Limitations table to reflect this.

Alternatively, if you want the 9:30 AM bar, change `scheduler.py`'s market-hours check to
use `datetime.minute` alongside `datetime.hour` and trigger at 10:30 when the first full bar
since open is confirmed closed.

---

### Fix 3 — Adjust Indicator Parameters for Hourly Timeframe

**Problem**

The MACD parameters (`MACD_FAST=12`, `MACD_SLOW=26`, `MACD_SIGNAL=9`) were designed for
daily charts. On hourly data 12 bars = 1.5 trading days and 26 bars = 3.3 trading days. The
"slow" EMA captures less than a week of data. The standard MACD crossover signal on hourly
data becomes noise-driven and fires many false signals per day.

The SMA and Bollinger window of 20 hourly bars = roughly 2.5 trading days. Mean reversion
and volatility signals over 2.5 days have no statistical significance for a stock at this
frequency.

**Why it matters**

Using daily-calibrated parameters on hourly data produces signals that fire many times per
session, driving overtrading. The Q-learner sees highly correlated states that appear
meaningful but are just noise.

**Fix**

In `config.py`, update these values:

```python
# SMA and Bollinger — 1 week of hourly bars (5 trading days × 6.5 hours ≈ 32 bars)
SMA_WINDOW: int = 40        # ~1 week of hourly bars (was 20)
BBANDS_WINDOW: int = 40     # ~1 week of hourly bars (was 20)

# MACD — scale to approximate daily behaviour on hourly data
# Daily 12/26/9 corresponds roughly to hourly 60/130/45
# A practical hourly setting that avoids overtrading:
MACD_FAST: int = 24         # ~3 trading days  (was 12)
MACD_SLOW: int = 52         # ~6.5 trading days (was 26)
MACD_SIGNAL: int = 18       # ~2.25 trading days (was 9)
```

After changing these values, force a full retrain (`python main.py --retrain`). The bin edges
in the saved Q-table will be stale and must be recalculated.

---

## Part 2 — Model Quality Improvements

These changes meaningfully improve decision quality and backtest reliability. All are necessary
for hourly trading; none should be skipped.

---

### Improvement 1 — Add RSI as a Fourth Indicator

**Problem**

The existing three indicators (SMA ratio, Bollinger %B, MACD) all measure slightly different
things but share a common weakness: they respond slowly to sharp reversals. RSI (Relative
Strength Index) provides a fast-responding overbought/oversold signal (0–100 scale) that
complements all three existing indicators and is specifically effective at hourly cadence.

**Why it matters**

For hourly trading, RSI captures intraday momentum exhaustion that MACD misses (MACD is a
lagging indicator). RSI and Bollinger %B together provide both a price-envelope signal and a
momentum-exhaustion signal. The Q-learner will have a richer state space to distinguish
trending from mean-reverting conditions.

**Fix**

In `indicators/technical.py`, add a new class:

```python
class RSIPredictor(BasePredictor):
    """
    Relative Strength Index: measures momentum via average gain vs average loss
    over a rolling window. Values near 100 = overbought; near 0 = oversold.
    """

    def compute(self, prices: pd.Series) -> pd.Series:
        """
        Compute RSI(RSI_WINDOW) for every bar.

        Uses Wilder's smoothing (ewm with alpha=1/window, adjust=False) to
        match the canonical RSI definition. NaN values at the start are
        forward-filled.

        Args:
            prices: Close price Series with a DatetimeIndex.

        Returns:
            Series of RSI values in [0, 100], same length as prices.
        """
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        alpha = 1.0 / config.RSI_WINDOW
        avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, float("inf"))
        rsi = 100 - (100 / (1 + rs))
        return rsi.ffill()

    def signal(self, value: float) -> int:
        """
        Map an RSI value to a discrete vote.

        Thresholds: RSI < 35 → bullish (oversold);
                    RSI > 65 → bearish (overbought);
                    otherwise → neutral.

        Args:
            value: A single RSI float in [0, 100].

        Returns:
            +1, -1, or 0.
        """
        if value < 35:
            return 1
        if value > 65:
            return -1
        return 0
```

In `config.py`, add:

```python
RSI_WINDOW: int = 14    # Standard RSI period; 14 hourly bars ≈ 2 trading days
```

In `main.py`, add `RSIPredictor` to the predictors list:

```python
from indicators.technical import (
    BollingerPercentBPredictor,
    MACDHistogramPredictor,
    RSIPredictor,
    SMARatioPredictor,
)

predictors = [
    SMARatioPredictor(),
    BollingerPercentBPredictor(),
    MACDHistogramPredictor(),
    RSIPredictor(),
]
```

**Important:** Adding a fourth indicator changes the state space from `3 × 8³ = 1,536` to
`3 × 8⁴ = 12,288` states. The Q-table size grows 8×. You must retrain after this change.
The `learner = QLearner(num_states=encoder.num_states(len(predictors)), ...)` line in
`main.py` already handles the new size automatically, but the existing `qtable.pkl` file
is incompatible and must be deleted (`python main.py --retrain`).

Consider also increasing `NUM_BINS` from 8 to 6 when using 4 indicators, to keep the state
space manageable. `3 × 6⁴ = 3,888` states is still well within training data coverage:

```python
NUM_BINS: int = 6   # Reduce from 8 when using 4 indicators; 3 × 6^4 = 3,888 states
```

---

### Improvement 2 — Fix Neutral Signal Accuracy Recording

**Problem**

In `model/predictor_manager.py`, the `record_accuracy()` method records a neutral vote (0)
as incorrect (accuracy = 0.0). Over time, the weight updater penalizes any predictor that
abstains, driving all predictors toward always taking a stance (+1 or -1) to preserve their
weight. This causes overtrading: predictors learn to vote loud rather than vote right.

**Why it matters for hourly trading**

Hourly markets are sideways 60–70% of the time between meaningful moves. A healthy model
should abstain (vote 0) when the signal is unclear. Penalizing abstention systematically
trains the weight system to ignore the signal quality and focus only on directional votes.

**Fix**

In `model/predictor_manager.py`, change the `record_accuracy` method to treat a neutral
vote as a 0.5 (neither rewarded nor penalized):

```python
def record_accuracy(self, prices: pd.Series, realized_return: float) -> None:
    for p in self._predictors.values():
        val = float(p.compute(prices).iloc[-1])
        vote = p.signal(val)
        if vote == 0:
            correct = 0.5  # abstention — no information, no penalty
        else:
            correct = 1.0 if (vote > 0) == (realized_return > 0) else 0.0
        p.record_accuracy(correct)
```

---

### Improvement 3 — Add Walk-Forward Validation to the Backtest

**Problem**

The current backtest uses a single train/test split: train on the earliest 510 days, test on
the last 90 days. A single split is fragile — the model's performance on that specific 90-day
window may not reflect its actual generalization ability. The 90-day window chosen happened to
be recent, which has look-ahead characteristics in hyperparameter selection (the researcher can
keep tweaking config until that window looks good).

**Why it matters**

Walk-forward validation (also called anchored walk-forward or rolling-window validation) divides
the historical data into multiple train/test folds. Each fold trains on everything before a
cutoff and tests on the next N days. This gives multiple independent performance samples and
a much more honest measure of whether the model generalizes.

**Fix**

In `backtest/runner.py`, add a `run_walk_forward` function alongside the existing `run_backtest`:

```python
def run_walk_forward(
    engine,
    full_start: str,
    full_end: str,
    test_window_days: int = 45,
    n_folds: int = 4,
) -> list[dict]:
    """
    Run anchored walk-forward validation across n_folds folds.

    For each fold:
      - Training window: full_start → fold_test_start
      - Test window:     fold_test_start → fold_test_start + test_window_days

    Args:
        engine: TradingEngine (will be retrained for each fold).
        full_start: Start of all available data "YYYY-MM-DD".
        full_end: End of all available data "YYYY-MM-DD".
        test_window_days: How many days each test window covers.
        n_folds: Number of folds to run.

    Returns:
        List of metric dicts, one per fold, each matching run_backtest() output.
    """
    from datetime import date, timedelta
    results = []
    end = date.fromisoformat(full_end)
    for fold in range(n_folds, 0, -1):
        test_end = end - timedelta(days=(fold - 1) * test_window_days)
        test_start = test_end - timedelta(days=test_window_days)
        train_end = test_start
        train_start = date.fromisoformat(full_start)
        if train_start >= train_end:
            continue
        print(f"\n[WF] Fold {n_folds - fold + 1}/{n_folds} "
              f"| Train: {train_start} → {train_end} | Test: {test_start} → {test_end}")
        engine.retrain(train_start.isoformat(), train_end.isoformat())
        metrics = run_backtest(engine, test_start.isoformat(), test_end.isoformat())
        metrics["fold"] = n_folds - fold + 1
        results.append(metrics)
    # Print summary across all folds
    avg_return = sum(r["cumulative_return"] for r in results) / len(results)
    avg_benchmark = sum(r["benchmark_cumulative_return"] for r in results) / len(results)
    print(f"\n[WF] Average return across {n_folds} folds: {avg_return:+.2f}%")
    print(f"[WF] Average benchmark across {n_folds} folds: {avg_benchmark:+.2f}%")
    return results
```

Add a `--walk-forward` CLI flag in `main.py` that calls this function instead of `run_backtest`.
Use the walk-forward average return (not the single-fold result) to assess whether changes to
indicators or hyperparameters actually improve the model.

---

### Improvement 4 — Add Sharpe Ratio and Max Drawdown to Backtest Metrics

**Problem**

The current backtest reports cumulative return, mean daily return, and std of daily return.
It does not report Sharpe ratio (risk-adjusted return) or maximum drawdown (worst peak-to-trough
loss). Cumulative return alone is not a useful measure for a trading model — a strategy that
makes 10% with 30% max drawdown is dramatically worse than one that makes 8% with 5% max
drawdown. For hourly trading specifically, drawdown control is critical.

**Fix**

In `backtest/runner.py`, update `run_backtest` to compute and report two additional metrics.
Add these computations before the `_print_metrics` call:

```python
# Sharpe ratio (annualised, assuming 252 trading days, 6.5 hours/day)
HOURLY_PERIODS_PER_YEAR = 252 * 6.5
sharpe = (mean_daily / std_daily * (HOURLY_PERIODS_PER_YEAR ** 0.5)
          if std_daily > 0 else 0.0)

# Maximum drawdown: largest peak-to-trough decline in portfolio value
if not daily_rets.empty:
    cumulative = (1 + daily_rets).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = float(drawdown.min())
else:
    max_drawdown = 0.0
```

Update `_print_metrics` to accept and print `sharpe` and `max_drawdown`. Update the returned
dict to include `"sharpe_ratio"` and `"max_drawdown_pct"` keys. Update the API's `/holdings`
route to expose these values.

---

### Improvement 5 — Slow Down Weight Decay for Hourly Adaptation

**Problem**

`WEIGHT_DECAY = 0.95` produces a half-life of roughly `log(0.5) / log(0.95) ≈ 14 steps`.
At hourly cadence during live trading, 14 steps = 14 hours ≈ 2 trading days. This means
the weight updater responds to individual noisy hours too quickly and can thrash indicator
weights based on a handful of false signals.

**Why it matters**

For hourly trading, a 2-day half-life is the right order of magnitude for detecting regime
changes (e.g. trending vs. mean-reverting). However, during backtesting there are thousands
of training bars, and a decay of 0.95 means early training accuracy almost entirely dominates
weight history. Set a value that gives a meaningful half-life during live operation.

**Fix**

In `config.py`, change:

```python
WEIGHT_DECAY: float = 0.97   # Half-life ~23 steps ≈ ~3 trading days at hourly cadence
```

This is a modest change but it stabilizes weight adaptation without making it unresponsive.

---

### Improvement 6 — Increase Early Stopping Patience

**Problem**

`EARLY_STOP_PATIENCE = 5` causes training to stop after 5 consecutive epochs without reward
improvement of at least `EARLY_STOP_DELTA = 1e-4`. With the ε-greedy exploration still active
(RAR decays slowly from 0.5), reward improvement per epoch is noisy. Five epochs is not enough
to distinguish "plateau" from "temporary noise." This causes cold starts to terminate in 10–15
epochs instead of fully utilizing the 50-epoch budget.

**Fix**

In `config.py`, change:

```python
EARLY_STOP_PATIENCE: int = 8     # Allow 8 quiet epochs before stopping (was 5)
EARLY_STOP_DELTA: float = 5e-5   # Lower improvement bar; reduce false stops (was 1e-4)
```

---

## Part 3 — Production Features (For When Hosted)

These features are needed before the bot runs unattended on a server.

---

### Feature 1 — Email Notification System

Add an email notification module that sends structured summaries to the operator.

**Where to add it:** Create `notifications/emailer.py`.

**Required content:**

```python
# notifications/emailer.py
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import config

def send_email(subject: str, body: str) -> None:
    """
    Send a plain-text email via SMTP to the configured recipient.

    Uses SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, and NOTIFY_EMAIL
    from config.py. All values default to empty strings so the function
    is a no-op when email is not configured, preventing crashes on
    environments without credentials.

    Args:
        subject: Email subject line.
        body: Plain-text email body.
    """
    if not config.SMTP_USER or not config.NOTIFY_EMAIL:
        return
    msg = MIMEMultipart()
    msg["From"] = config.SMTP_USER
    msg["To"] = config.NOTIFY_EMAIL
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))
    try:
        with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT) as server:
            server.starttls()
            server.login(config.SMTP_USER, config.SMTP_PASS)
            server.send_message(msg)
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("send_email failed: %s", exc)
```

**Add to `config.py`:**

```python
# --- Email notifications ---
SMTP_HOST: str = ""          # e.g. "smtp.gmail.com" — leave blank to disable email
SMTP_PORT: int = 587         # Standard STARTTLS port
SMTP_USER: str = ""          # Sender email address / SMTP login
SMTP_PASS: str = ""          # SMTP password or app-specific password
NOTIFY_EMAIL: str = ""       # Recipient email for notifications
```

**Trigger points** — call `send_email()` from:

1. `engine/trading_engine.py` — at the end of `train()`, send a training summary:
   `subject="Bot Training Complete", body=f"Epochs: {e}, Reward: {best_reward:.4f}"`

2. `engine/trading_engine.py` — at the end of `run_cycle()`, send a cycle summary with
   portfolio value, each trade taken this cycle, and current indicator weights.

3. `backtest/runner.py` — at the end of `run_backtest()`, send the full metrics block.

4. Any `logger.error()` call — also send an email so critical failures are immediately visible.

**Recommended email service options:**

| Service | Free tier | Setup |
|---|---|---|
| Gmail (App Password) | Unlimited personal | Enable 2FA, generate App Password, use `smtp.gmail.com:587` |
| SendGrid | 100 emails/day free | Sign up, get API key, use `smtp.sendgrid.net:587` |
| Mailgun | 1,000 emails/month free | Sign up, get SMTP credentials, use `smtp.mailgun.org:587` |
| AWS SES | 62,000 emails/month if hosted on AWS EC2 | Verify sender domain, use SMTP endpoint |

For a personal trading bot sending a few emails per day, Gmail App Password is the simplest.
For production or more volume, SendGrid or Mailgun are more reliable.

---

### Feature 2 — Market Holiday Awareness

**Problem**

The scheduler in `engine/scheduler.py` only checks that the current time is a weekday between
`MARKET_OPEN_HOUR` and `MARKET_CLOSE_HOUR`. US markets are closed on roughly 9 holidays per
year (New Year's, MLK Day, Presidents Day, Good Friday, Memorial Day, Juneteenth, Independence
Day, Labor Day, Thanksgiving, Christmas). Firing live cycles on these days wastes API calls and
may log spurious "no data" errors.

**Fix**

Add `pandas_market_calendars` to `requirements.txt`:

```
pandas_market_calendars>=4.3.1
```

In `engine/scheduler.py`, update `is_market_hours()`:

```python
import pandas_market_calendars as mcal

def is_market_hours(self) -> bool:
    """Return True if the US stock market is currently open."""
    import pytz
    from datetime import datetime
    nyse = mcal.get_calendar("NYSE")
    et = pytz.timezone("US/Eastern")
    now = datetime.now(tz=et)
    schedule = nyse.schedule(
        start_date=now.strftime("%Y-%m-%d"),
        end_date=now.strftime("%Y-%m-%d"),
    )
    if schedule.empty:
        return False  # holiday or weekend
    market_open = schedule.iloc[0]["market_open"].tz_convert(et)
    market_close = schedule.iloc[0]["market_close"].tz_convert(et)
    return market_open <= now <= market_close
```

---

## Part 4 — Hosting Options

The bot runs as a single Python process with:
- A background APScheduler thread (hourly cycles)
- A uvicorn FastAPI server (port 8000)
- File I/O to `logs/` directory

Any hosting option needs: a persistent process (not serverless), persistent disk for
`logs/qtable.pkl`, Python 3.11+, and outbound HTTPS to Yahoo Finance.

---

### Option A — Railway (Recommended for getting started)

**Cost:** Free tier includes 500 hours/month; ~$5/month for always-on  
**Setup time:** 15 minutes  
**Pros:** Automatic GitHub deploys, persistent disk, easy environment variables, no Docker required  
**Cons:** Free tier sleeps after inactivity; upgrade to paid to keep bot always-on

Steps:
1. Push repo to GitHub
2. Sign up at railway.app, create a new project from your GitHub repo
3. Set start command: `python main.py` (Railway auto-detects requirements.txt)
4. Add a Volume in Railway and set mount path to `/app/logs` for Q-table persistence
5. Set environment variables for email in the Railway dashboard (SMTP_USER, SMTP_PASS, etc.)

---

### Option B — Fly.io

**Cost:** Free tier (3 shared VMs); ~$2–5/month for a persistent VM  
**Setup time:** 30 minutes  
**Pros:** Fast deploys via `flyctl`, persistent volumes, global edge network  
**Cons:** Requires Dockerfile; slightly steeper learning curve than Railway

Steps:
1. Install `flyctl`, run `fly launch` in the repo directory
2. Create a `Dockerfile`:
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   CMD ["python", "main.py"]
   ```
3. Run `fly volumes create tradebot_logs --size 1` for persistence
4. Mount the volume in `fly.toml` at `/app/logs`
5. Set secrets: `fly secrets set SMTP_USER=... SMTP_PASS=...`

---

### Option C — DigitalOcean Droplet

**Cost:** $4–6/month (smallest droplet)  
**Setup time:** 45 minutes  
**Pros:** Full control, persistent disk included, cheap, no cold starts  
**Cons:** Manual setup; you manage the server

Steps:
1. Create an Ubuntu 22.04 droplet ($4/month, 1 GB RAM)
2. SSH in, install Python 3.11, clone the repo
3. Create a `.env` file for secrets (email credentials)
4. Run with `systemd` or `tmux` to keep it alive on reboot:
   ```ini
   # /etc/systemd/system/tradebot.service
   [Unit]
   Description=Trading Bot
   After=network.target
   [Service]
   WorkingDirectory=/home/ubuntu/tradebot
   ExecStart=/home/ubuntu/tradebot/.venv/bin/python main.py
   Restart=always
   [Install]
   WantedBy=multi-user.target
   ```
5. `systemctl enable tradebot && systemctl start tradebot`

---

### Option D — AWS EC2 (t3.micro — Free Tier)

**Cost:** Free for 12 months on new accounts; ~$8/month after  
**Setup time:** 1 hour  
**Pros:** Pairs with AWS SES for free email; reliable uptime; integrates with CloudWatch for logs  
**Cons:** Most setup work; overkill for a single bot

Use the same `systemd` approach as the DigitalOcean Droplet. If using AWS SES for email,
set `SMTP_HOST="email-smtp.us-east-1.amazonaws.com"` and verify your sender address in the
AWS Console.

---

### Option E — Local Machine (Development Only)

Run `python main.py` locally during market hours. Use Task Scheduler (Windows) or cron (Mac/Linux)
to start the script at 9:30 AM and kill it at 4 PM. Not reliable for production because your
machine must stay awake and connected.

---

## Part 5 — README Corrections

The following items in `README.md` are factually incorrect and should be updated.

---

### Correction 1 — Training Lookback Days

The Limitations table says:

> Maximum 730 days of training history — Yahoo Finance only provides hourly data that far back

`config.py` sets `TRAIN_LOOKBACK_DAYS = 600`. The practical limit in use is 600 days, not 730.
Update the Limitations table to say:

> Maximum 730 days of training history (currently configured to 600 days) — Yahoo Finance caps
> hourly interval data at 730 calendar days

---

### Correction 2 — Market Open Hour

The README says:

> (9 AM – 4 PM Eastern, Monday through Friday)

The US stock market opens at 9:30 AM Eastern, not 9:00 AM. `config.py` also has this wrong
(`MARKET_OPEN_HOUR = 9`). After applying Fix 2 above (changing to `MARKET_OPEN_HOUR = 10`),
update the README to say:

> (first cycle at 10 AM Eastern to ensure the 9:30 opening bar is complete, through 4 PM
> Eastern, Monday through Friday excluding market holidays)

---

### Correction 3 — Output Format in README Uses `Exploration` Label

The training output example in the README says:

```
[TRAIN] Epoch 1/50 | Reward: 0.0842 | Exploration: 50.0%
```

The actual code in `trading_engine.py` prints `rar` not `Exploration:`. Update the example to:

```
[TRAIN] Epoch 1/50 | Reward: 0.0842 | rar: 0.5000
```

---

## Summary of All Changes

| # | File(s) | Type | Impact |
|---|---|---|---|
| Fix 1 | `engine/trading_engine.py` | Bug fix | Aligns training with live behavior |
| Fix 2 | `config.py`, `README.md` | Bug fix | Prevents pre-market stale data cycles |
| Fix 3 | `config.py` | Parameter | Corrects indicators for hourly timeframe |
| Imp 1 | `indicators/technical.py`, `config.py`, `main.py` | New indicator | Adds RSI for overbought/oversold |
| Imp 2 | `model/predictor_manager.py` | Logic fix | Stops penalizing neutral abstentions |
| Imp 3 | `backtest/runner.py`, `main.py` | New function | Walk-forward validation |
| Imp 4 | `backtest/runner.py` | Metrics | Sharpe ratio and max drawdown |
| Imp 5 | `config.py` | Parameter | Stabilizes weight adaptation speed |
| Imp 6 | `config.py` | Parameter | Reduces premature early stopping |
| Feat 1 | `notifications/emailer.py`, `config.py` | New module | Email alerts for trades and errors |
| Feat 2 | `engine/scheduler.py`, `requirements.txt` | Enhancement | Skips market holidays correctly |
| Doc 1–3 | `README.md` | Corrections | Fixes factual errors |

---

## Implementation Order

Implement in exactly this order. After each group, run `pytest tests/ -v` and confirm no
regressions before proceeding.

**Group 1 (Critical — do first):** Fix 1, Fix 2, Fix 3  
**Group 2 (Retrain required):** Improvement 1 (RSI), then run `python main.py --retrain`  
**Group 3 (No retrain needed):** Improvements 2, 5, 6  
**Group 4 (Backtest tooling):** Improvements 3, 4  
**Group 5 (Production):** Feature 1 (Email), Feature 2 (Holidays)  
**Group 6 (Documentation):** All README corrections

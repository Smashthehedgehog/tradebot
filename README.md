# Trading Bot — What It Is, What It Does, and How It Works

---

## What Is This Bot?

This is an automated stock trading model that learns from historical market data,
makes hourly buy and sell decisions during market hours, and continuously improves
itself based on which of its instincts were right or wrong.

It does **not** connect to any brokerage yet. Instead of placing real orders, it
prints every decision it would have made to the screen — what it would buy, sell,
or hold, at what price, and why. This lets you watch it think and verify it is
behaving sensibly before ever touching real money.

---

## The Big Picture — What Happens When You Start It

```
Start the bot
     │
     ├─── Is there a saved brain on disk?
     │         │
     │    YES ─┴─ Load the saved brain and skip straight to live trading
     │
     NO ── Study the past 730 days of hourly stock data (from Yahoo Finance)
               │
               └─── Learn patterns → save its brain → run a practice test
                         │
                         └─── Start making hourly decisions during market hours
                                    │
                                    └─── Every hour: improve its own instincts
```

---

## The Three Phases

### Phase 1 — Studying (Training)

Before the bot does anything live, it reads roughly two years of past hourly
stock prices from Yahoo Finance. It studies each stock hour by hour and asks
itself: *"If I had been long, short, or doing nothing at this moment in time,
and I had seen these market signals, what should I have done to make the most
money?"*

It tries thousands of combinations, learning from its mistakes each time, until
it settles on a reliable playbook. This playbook is saved to disk so it does not
need to re-study every time you restart the bot.

### Phase 2 — Practice Test (Backtest)

After studying, the bot is tested on the final 90 days of that same data —
days it has never seen before. It trades as if those days were live and reports
its score: how much money it would have made or lost, how often it traded, and
how it compared to simply buying and holding the S&P 500 index for those 90 days.

### Phase 3 — Live Decisions (Every Hour)

Once the practice test is done, the bot wakes up every hour during market hours
(first cycle at 10 AM Eastern to ensure the 9:30 opening bar is complete, through
4 PM Eastern, Monday through Friday excluding market holidays). For each stock it watches, it
looks at the latest price data, runs it through its learned playbook, and decides
whether to buy, sell, or hold. Every decision is printed to the screen with a
full explanation.

---

## How the Bot Learns — Plain English

The bot uses a technique called **reinforcement learning** — the same family of
methods used to train chess engines and game-playing AIs. The idea is simple:

- The bot tries an action (buy, sell, or hold).
- It sees what happened next (did the price go up or down?).
- If the action was profitable, it remembers to favor that action in similar
  situations. If it was costly, it learns to avoid it.
- Over thousands of practice rounds, the bot builds a lookup table — a
  "cheat sheet" — that maps *"I am in this situation"* to *"here is the best
  thing to do."*

A **situation** is defined by three things:
1. What market signals the bot is currently seeing (explained below).
2. Whether the bot is currently holding shares, shorting, or sitting flat.

---

## The Three Market Signals (Indicators)

The bot watches three independent signals to understand what the market is doing.
Think of each one as a different expert giving their opinion.

### Signal 1 — Price vs. Average (SMA Ratio)
> *"Is this stock unusually cheap or unusually expensive compared to its recent
> average?"*

The bot computes a 20-hour rolling average of the stock price and compares the
current price to it. If the price is significantly below average, the expert
votes **Buy** (mean-reversion opportunity). If significantly above, it votes
**Sell**. Otherwise, it stays neutral.

### Signal 2 — Bollinger Band Position (%B)
> *"Is this stock near the edge of its normal volatility range?"*

The bot tracks upper and lower "bands" that represent normal price swings. If
the price touches or breaks below the lower band, this expert votes **Buy**
(oversold). If it touches the upper band, it votes **Sell** (overbought).

### Signal 3 — Momentum (MACD Histogram)
> *"Is the price gaining speed upward or downward?"*

This signal compares two moving averages of different speeds to detect momentum
shifts. When short-term momentum is stronger than long-term momentum, this expert
votes **Buy**. When it weakens, it votes **Sell**.

---

## Self-Improvement — How the Bot Tunes Itself

Each of the three signals has a **weight** — essentially how much influence that
expert's opinion carries in the final decision. Weights start equal (one-third
each), but the bot adjusts them automatically every hour based on recent
performance:

- After each hour, the bot checks whether each signal's vote matched what
  actually happened in the market (did the price move in the direction the
  signal predicted?).
- Signals that have been consistently right recently get a higher weight.
- Signals that have been wrong recently get a lower weight (but never zero —
  every signal always has a small minimum voice).
- This means the bot can adapt over time. If, for example, momentum stops
  working during a flat, sideways market, its weight will naturally shrink
  until conditions change.

You can also **manually override a signal's weight** at any time through the
API (for example, if you know MACD tends to be misleading in a specific market
environment).

---

## Files and What They Each Do

### `config.py` — The Control Panel
All settings in one place. Want to trade different stocks? Change them here.
Want longer training? Adjust here. No digging through code required.

Key settings:
- Which stocks to trade
- How much starting cash
- How many shares to buy per order
- How far back to study (up to 730 days)
- How many days to hold back for practice testing (default 90)

---

### `data/fetcher.py` — The Market Data Collector
Downloads stock price history from Yahoo Finance. It knows how to:
- Grab a full history of hourly prices for a date range (used during studying)
- Grab just the most recent completed hour of prices (used during live trading)
- Handle errors gracefully — if a symbol fails to download, it logs a warning
  and moves on rather than crashing everything

---

### `data/cache.py` — The Short-Term Memory
Saves downloaded price data in memory (and to disk as a backup) so the bot does
not re-download the same data repeatedly. When the bot restarts, it reloads the
cached data from disk automatically.

---

### `indicators/base.py` — The Signal Template
Defines the rules every signal must follow: it must be able to read a price
history and produce a continuous score, and it must be able to translate that
score into a simple vote (Buy/Sell/Neutral). Every signal in the system follows
this contract.

---

### `indicators/technical.py` — The Three Experts
Contains the actual code for the SMA Ratio, Bollinger Band, and MACD signals
described above. Each one reads price history and produces its vote. Adding a
new signal (e.g. news sentiment) means adding a new class here — nothing else
needs to change.

---

### `model/predictor_manager.py` — The Voting Coordinator
Holds all three signals and manages their weights. When a decision needs to be
made, it asks each signal for its vote, multiplies each vote by its weight, and
combines them into a single number between -1 (strong sell) and +1 (strong buy).
It also records whether each signal was right or wrong after each hour, which
feeds into the self-improvement system.

---

### `model/weight_updater.py` — The Performance Tracker
After each hour, this component reviews whether each signal's prediction matched
reality. It uses a slow-moving average of accuracy (recent hours matter more than
old ones) to calculate updated weights and writes them back to the coordinator.
This is the self-improvement loop.

---

### `model/state_encoder.py` — The Situation Summarizer
Takes the three continuous signal values and the bot's current position
(holding, shorting, or flat) and compresses them into a single code number. That
number is used to look up the best action in the bot's learned playbook. During
training, it also figures out the "bins" (ranges of values) that group similar
market situations together — like sorting temperatures into cold, mild, warm, hot.

---

### `model/q_learner.py` — The Brain / Playbook
The core learning engine. Maintains a large table with one row per possible
situation and three columns (one per action: Hold, Buy, Sell). Each cell holds
a score representing *"how good is it to take this action in this situation?"*

During **studying**, these scores are updated thousands of times based on what
actually would have happened. During **live trading**, the bot simply reads the
highest score in the current situation's row and takes that action. The table
is saved to disk after training so it survives restarts.

---

### `portfolio/tracker.py` — The Simulated Wallet
Keeps track of how much cash the bot has and how many shares of each stock it
holds — all simulated, no real money moved. Every time the bot decides to buy or
sell, this component:
1. Checks the decision is valid (enough cash? enough shares to sell?)
2. Updates the simulated cash and holdings
3. Records the trade in a full history log
4. **Prints the decision to the screen** in a readable format

Every decision, even a Hold, is printed so you always know what the bot is
thinking.

---

### `engine/trading_engine.py` — The Manager
The central coordinator that runs everything in the right order. It:
- Manages the studying phase (calls the data collector, signals, and brain)
- Manages the practice test
- Runs the hourly live decision cycle
- Handles loading a saved brain from disk (warm start) vs. re-studying from
  scratch (cold start)
- Handles retraining on demand

---

### `engine/scheduler.py` — The Alarm Clock
Uses a background timer to wake the bot up every 60 minutes. Before running a
decision cycle, it checks whether the market is currently open (weekdays,
9 AM – 4 PM Eastern). If the market is closed, it goes back to sleep without
doing anything.

---

### `backtest/runner.py` — The Practice Exam Grader
Replays the bot's decisions over the held-out 90-day test window (data the bot
never studied) and measures how well it would have done. It reports:
- Total return vs. the S&P 500 benchmark
- Average daily gain/loss
- Volatility of daily returns
- Total number of trades made
- Final portfolio value

---

### `api/server.py` — The Status Window
Runs a small web server on your computer (port 8000) that any other program —
like a React dashboard — can ask questions of. Available at any time while the
bot is running:

| What you can ask | What you get back |
|---|---|
| Is the bot trained and is the market open? | Yes/No status |
| What is the portfolio worth right now? | Cash, holdings, profit/loss |
| Which signals are most trusted right now? | Current weights and accuracy scores |
| What trades has the bot made so far? | Full history, filterable by stock |
| Retrain the bot on a new date range | Starts retraining in the background |
| Manually adjust a signal's weight | Applies immediately |

---

### `main.py` — The On Switch
The single file you run to start everything. In order, it:
1. Reads any command-line options you pass
2. Builds all the components listed above
3. Calculates the training and test date windows from today's date
4. Either loads a saved brain (fast) or studies from scratch (slower, first time)
5. Runs the practice test and prints the score
6. Starts the hourly alarm clock
7. Starts the web API

---

## What Gets Printed to the Screen

During **studying**, you see progress updates every epoch (training round):
```
[TRAIN] Starting historical pre-training: 2024-04-22 → 2026-01-26
[TRAIN] Fetched 4,820 hourly bars for AAPL
[TRAIN] Epoch 1/50 | Reward: 0.0842 | rar: 0.5000
[TRAIN] Epoch 2/50 | Reward: 0.1203 | rar: 0.4990
...
[TRAIN] Training complete. Epochs: 38 | Final reward: 0.4871 | Elapsed: 14.2s
```

During **live trading**, every hourly decision prints like this:
```
[TRADE] 2026-04-27 10:00:00 ET | AAPL  | BUY  | 10 shares @ $213.45 | Cash: $97,865.50 | Reason: Weighted signal +0.72 (SMA:+1 BB:+1 MACD:0)
[TRADE] 2026-04-27 10:00:00 ET | MSFT  | HOLD |  0 shares @ $425.10 | Cash: $97,865.50 | Reason: Weighted signal +0.05 (SMA:0 BB:+1 MACD:-1)
[TRADE] 2026-04-27 10:00:00 ET | GOOG  | SELL | 10 shares @ $178.90 | Cash: $97,865.50 | Reason: Weighted signal -0.61 (SMA:-1 BB:-1 MACD:0)
[CYCLE] 2026-04-27 10:00:00 ET | Portfolio: $101,234.80
```

Every line is structured consistently (fields separated by `|`) so a future
dashboard can parse and display it automatically.

---

## How to Add a New Signal

The system is designed so adding a new signal (for example, one based on news
headlines or earnings reports) requires touching only two things:

1. Write one new Python class in `indicators/technical.py` that reads a price
   history and produces a vote.
2. Add one line to `main.py` to include it in the list of signals.

The coordinator, weight system, brain, and API all pick it up automatically.
You then retrain, and the bot will learn how much to trust the new signal
relative to the existing three.

---

## Limitations to Know About

| Limitation | Why |
|---|---|
| Maximum 730 days of training history (configured at 600 days) | Yahoo Finance caps hourly interval data at 730 calendar days |
| Training takes a few minutes on first run | The bot studies thousands of hourly bars across many stocks |
| No real trades are placed | Brokerage connection is not built yet |
| The bot does not read news or earnings | Only price-based signals at this stage |
| Retraining is needed when you add new signals | The brain's lookup table must be rebuilt |

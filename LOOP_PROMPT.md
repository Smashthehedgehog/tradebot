# Loop Task: Run Until Backtest Beats Benchmark

## Objective

Repeatedly run `main.py` using the venv Python interpreter until the backtest output shows the model's cumulative return is **strictly greater than** the benchmark (S&P 500) cumulative return. Once that condition is met, stop and report the final results.

---

## Environment

- **Working directory:** `c:\Users\mikeani\Documents\GitHub\tradebot`
- **Python interpreter:** `.venv\Scripts\python.exe`
- **Entry point:** `main.py`
- **Run command:** `.venv\Scripts\python.exe main.py --retrain`

Use `--retrain` on every run to force a full cold-start retrain and get a fresh backtest result each time.

---

## How to Read the Backtest Output

After training completes, the script prints a results block to stdout in this format:

```
=======================================================
  BACKTEST RESULTS
=======================================================
  Final portfolio value :   $xxx,xxx.xx
  Cumulative return     :      +x.xx%
  Benchmark (S&P 500)   :      +x.xx%
  Mean daily return     :      +x.xxxx%
  Std of daily return   :       x.xxxx%
  Total trades          :          xxx
=======================================================
```

Parse the two lines:
- `Cumulative return` → model return
- `Benchmark (S&P 500)` → benchmark return

---

## Success Condition

**Stop looping when:**

```
Cumulative return > Benchmark (S&P 500)
```

Both values are percentages — compare them numerically. For example, `+3.21%` is greater than `+1.87%`.

---

## Failure / Error Handling

- If the script crashes or prints a training failure message (e.g. `no price data fetched`), wait 30 seconds and retry.
- If `main.py` starts the FastAPI server and does not exit on its own, terminate the process after the backtest block is printed — do not wait for the server.
- Cap total attempts at **20 runs**. If the condition is never met after 20 runs, stop and report the best result seen across all runs.

---

## Adding Stocks After Repeated Failure

If after **10 consecutive runs** the model has not significantly beaten the benchmark (i.e. cumulative return is not at least **2 percentage points above** the benchmark), expand the symbol universe by adding 2 more stocks:

1. Edit `config.py` and append 2 new well-known, liquid ticker symbols to the `SYMBOLS` list. Choose symbols not already in the list — good candidates are large-cap US equities such as `"AMZN"`, `"NVDA"`, `"META"`, `"TSLA"`, or `"JPM"`.
2. Save `config.py`.
3. Reset the run counter back to 1 and continue looping for up to another 20 runs with the expanded symbol list.
4. Only add stocks once — do not add more symbols on a second round of failures.

After adding stocks, print a note in your final report indicating which symbols were added and on which run the change was made.

---

## What to Report When Done

When the success condition is met (or the cap is reached), report:

1. Which run number succeeded (e.g. "Run 4 of 20")
2. The full backtest results block from that run
3. Whether the cap was hit without success, and if so, the best cumulative return seen and which run it came from

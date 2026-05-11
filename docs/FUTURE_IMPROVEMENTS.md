# Future Improvements

## Dynamic Symbol Management

Currently `SYMBOLS` is a static list in `config.py`. A future improvement would
allow symbols to be added and removed at runtime without restarting the bot.

### What needs to change

**Adding a symbol at runtime**
- Expose a `POST /symbols` API route that accepts a new ticker
- `TradingEngine` appends the symbol to `self.symbols` and initialises a zero
  holding in `PortfolioTracker`
- Fetch historical price data for the new symbol and populate the cache
- The existing Q-table and encoder bin edges can be reused immediately — the
  Q-learner operates on indicator values, not symbol identities, so no retraining
  is required to start making decisions for the new symbol
- Optionally trigger a background retrain so the new symbol's patterns are
  incorporated into the Q-table

**Removing a symbol at runtime**
- Expose a `DELETE /symbols/{symbol}` API route
- Force-close any open position before removing (execute a SELL if shares are held)
- Remove the symbol from `self.symbols` and the tracker's holdings dict
- No retraining required

**Persisting the symbol list across restarts**
- Save the current symbol list to `logs/symbols.json` whenever it changes
- On startup, load from `logs/symbols.json` if it exists, falling back to
  `config.SYMBOLS` if not

### Considerations
- Adding many symbols increases the training time proportionally — a retrain
  after adding several symbols at once is more efficient than one retrain per symbol
- Symbols with very low volume or incomplete yfinance hourly data should be
  validated before being added (check that `fetch_prices` returns at least
  `SMA_WINDOW` bars)
- The portfolio tracker's `holdings` dict should handle unknown symbols gracefully
  rather than raising a KeyError

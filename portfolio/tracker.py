import logging
from datetime import datetime, timezone

import pandas as pd

import config

logger = logging.getLogger(__name__)


class PortfolioTracker:
    """
    Simulates cash and share holdings for every trade decision.

    No real orders are placed. Every action (BUY, SELL, HOLD) is validated,
    recorded in a full history log, and printed to stdout in a structured
    format so the frontend can later parse it.
    """

    def __init__(self, starting_cash: float, symbols: list[str]) -> None:
        """
        Initialise the simulated wallet with cash and zero shares.

        Args:
            starting_cash: Starting cash balance in USD.
            symbols: List of ticker symbols; each initialised to 0 shares held.
        """
        self.starting_cash: float = starting_cash
        self.cash: float = starting_cash
        self.holdings: dict[str, int] = {s: 0 for s in symbols}
        self.history: list[dict] = []

    def execute(
        self,
        symbol: str,
        action: str,
        shares: int,
        price: float,
        reason: str = "",
    ) -> None:
        """
        Simulate one trade, update cash and holdings, and print the result.

        For BUY: deducts cost including simulated market impact and commission.
        For SELL: adds proceeds after impact and commission deductions.
        For HOLD: no cash or holdings change; still printed and recorded.

        Invalid trades (insufficient cash or shares) are skipped with a WARNING
        and are not appended to history.

        Args:
            symbol: Ticker symbol being traded, e.g. "AAPL".
            action: One of "BUY", "SELL", or "HOLD".
            shares: Number of shares involved in the trade (0 for HOLD).
            price: Current price per share in USD.
            reason: Human-readable explanation printed alongside the decision.
        """
        if action == "BUY" and shares > 0:
            cost = shares * price * (1.0 + config.IMPACT) + config.COMMISSION
            if cost > self.cash:
                logger.warning(
                    "execute: BUY skipped for %s — need $%.2f, have $%.2f",
                    symbol, cost, self.cash,
                )
                return
            self.cash -= cost
            self.holdings[symbol] = self.holdings.get(symbol, 0) + shares

        elif action == "SELL" and shares > 0:
            held = self.holdings.get(symbol, 0)
            if shares > held:
                logger.warning(
                    "execute: SELL skipped for %s — need %d shares, hold %d",
                    symbol, shares, held,
                )
                return
            proceeds = shares * price * (1.0 - config.IMPACT) - config.COMMISSION
            self.cash += proceeds
            self.holdings[symbol] = held - shares

        ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        record = {
            "timestamp": ts,
            "symbol": symbol,
            "action": action,
            "shares": shares,
            "price": round(price, 4),
            "cash_after": round(self.cash, 4),
            "holdings_snapshot": dict(self.holdings),
            "reason": reason,
        }
        self.history.append(record)

        print(
            f"[TRADE] {ts} ET | {symbol:<6} | {action:<4} | {shares:>5} shares"
            f" @ ${price:>10.2f} | Cash: ${self.cash:>12,.2f} | Reason: {reason}"
        )

    def portfolio_value(self, current_prices: dict[str, float]) -> float:
        """
        Compute the total simulated portfolio value at current market prices.

        Args:
            current_prices: Dict mapping symbol to its current price per share.

        Returns:
            Cash balance plus the market value of all held shares.
        """
        stock_value = sum(
            self.holdings.get(sym, 0) * current_prices.get(sym, 0.0)
            for sym in self.holdings
        )
        return self.cash + stock_value

    def daily_returns(self, price_history: dict[str, pd.Series]) -> pd.Series:
        """
        Reconstruct a daily portfolio value time series from trade history and
        return its percent change (daily returns).

        At each date in the price_history index, the tracker replays all trades
        up to that date to determine the cash and holdings state, then computes
        portfolio value using the provided prices.

        Args:
            price_history: Dict mapping symbol to a Close price Series indexed
                           by date. All series should share the same DatetimeIndex.

        Returns:
            Daily percent-change Series; empty if history or prices are absent.
        """
        if not self.history or not price_history:
            return pd.Series(dtype=float)

        first_series = next(iter(price_history.values()))
        dates = first_series.index

        sorted_history = sorted(self.history, key=lambda x: x["timestamp"])
        history_idx = 0
        cash = self.starting_cash
        holdings: dict[str, int] = {s: 0 for s in self.holdings}

        portfolio_values = []
        for date in dates:
            while history_idx < len(sorted_history):
                trade_ts = pd.Timestamp(sorted_history[history_idx]["timestamp"])
                if trade_ts.tzinfo is None:
                    trade_ts = trade_ts.tz_localize("UTC")
                if trade_ts.normalize() <= date.normalize():
                    cash = sorted_history[history_idx]["cash_after"]
                    holdings = dict(sorted_history[history_idx]["holdings_snapshot"])
                    history_idx += 1
                else:
                    break
            value = cash + sum(
                holdings.get(sym, 0) * price_history[sym].get(date, 0.0)
                for sym in holdings
            )
            portfolio_values.append(value)

        values = pd.Series(portfolio_values, index=dates)
        return values.pct_change().dropna()

    def summary(self, current_prices: dict[str, float]) -> dict:
        """
        Return a snapshot of the portfolio suitable for the API's GET /holdings route.

        Args:
            current_prices: Dict mapping symbol to its current price per share.

        Returns:
            Dict with keys: total_value, cash, holdings, pnl, pnl_pct, num_trades.
        """
        total = self.portfolio_value(current_prices)
        pnl = total - self.starting_cash
        num_trades = sum(1 for h in self.history if h["action"] != "HOLD")
        return {
            "total_value": round(total, 2),
            "cash": round(self.cash, 2),
            "holdings": dict(self.holdings),
            "pnl": round(pnl, 2),
            "pnl_pct": round((pnl / self.starting_cash) * 100, 4) if self.starting_cash else 0.0,
            "num_trades": num_trades,
        }

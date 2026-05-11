import logging
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config
from data.cache import price_cache
from data.fetcher import fetch_multiple, fetch_latest_bar
from model.predictor_manager import PredictorManager
from model.weight_updater import WeightUpdater
from model.state_encoder import StateEncoder
from model.q_learner import QLearner
from portfolio.tracker import PortfolioTracker

logger = logging.getLogger(__name__)

# Map holding shares -> holding index used by the state encoder
_HOLDING_IDX = {True: 1, False: 0}  # long=1, flat/short=0 (simplified for live)


class TradingEngine:
    """
    Central orchestrator that wires together all model components.

    Responsible for the full training pipeline (cold start), loading a saved
    model (warm start), running one hourly decision cycle, and triggering
    retraining on demand.
    """

    def __init__(
        self,
        symbols: list[str],
        manager: PredictorManager,
        encoder: StateEncoder,
        learner: QLearner,
        tracker: PortfolioTracker,
        updater: WeightUpdater,
    ) -> None:
        """
        Initialise the engine with all pre-built components.

        Args:
            symbols: List of ticker symbols to trade.
            manager: PredictorManager holding all active indicators.
            encoder: StateEncoder for discretising indicator values.
            learner: QLearner containing the Q-table.
            tracker: PortfolioTracker for simulated cash and holdings.
            updater: WeightUpdater for online predictor weight adjustment.
        """
        self.symbols = symbols
        self.manager = manager
        self.encoder = encoder
        self.learner = learner
        self.tracker = tracker
        self.updater = updater
        self.is_trained: bool = False
        self._prev_prices: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, train_start: str, train_end: str) -> None:
        """
        Run the full historical pre-training pipeline using hourly yfinance data.

        Steps:
          1. Fetch TRAIN_INTERVAL hourly bars for all symbols.
          2. Compute all indicator time series for each symbol.
          3. Fit the StateEncoder on all indicator data combined.
          4. Run the Q-learning epoch loop with early stopping.
          5. Save the Q-table and bin edges to QTABLE_PATH.
          6. Run one weight_updater pass using accumulated accuracy.

        Args:
            train_start: Start date string "YYYY-MM-DD" (inclusive).
            train_end: End date string "YYYY-MM-DD" (exclusive).
        """
        print(f"[TRAIN] Starting historical pre-training: {train_start} -> {train_end}")
        t0 = time.time()

        # Step 1: Fetch price data
        price_data = fetch_multiple(self.symbols, train_start, train_end, config.TRAIN_INTERVAL)
        if not price_data:
            logger.error("train: no price data fetched — aborting training")
            return

        for sym in price_data:
            print(f"[TRAIN] Fetched {len(price_data[sym])} hourly bars for {sym}")
            price_cache.put(sym, config.TRAIN_INTERVAL, price_data[sym])

        # Step 2: Compute indicators for each symbol
        indicator_data: dict[str, pd.DataFrame] = {}
        for sym, df in price_data.items():
            prices = df["Close"]
            ind_dict = {
                name: predictor.compute(prices)
                for name, predictor in self.manager.predictors().items()
            }
            indicator_data[sym] = pd.DataFrame(ind_dict).dropna()

        # Step 3: Fit encoder on all training indicator values combined
        all_indicators = pd.concat(list(indicator_data.values()), axis=0)
        self.encoder.fit(all_indicators)
        column_order = list(all_indicators.columns)

        # Step 4: Q-learning epoch loop
        best_reward = -np.inf
        patience_counter = 0

        for epoch in range(1, config.MAX_EPOCHS + 1):
            epoch_reward = self._run_epoch(price_data, indicator_data)
            improvement = epoch_reward - best_reward

            print(
                f"[TRAIN] Epoch {epoch}/{config.MAX_EPOCHS}"
                f" | Reward: {epoch_reward:.4f}"
                f" | rar: {self.learner.rar:.4f}"
            )

            if improvement > config.EARLY_STOP_DELTA:
                best_reward = epoch_reward
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.EARLY_STOP_PATIENCE:
                    print(f"[TRAIN] Early stop at epoch {epoch} — reward stable")
                    break

        # Step 5: Save Q-table + bin edges
        self.learner.save(config.QTABLE_PATH, self.encoder.bin_edges, column_order)

        # Step 6: Initial weight update pass
        self.updater.update()

        self.is_trained = True
        elapsed = time.time() - t0
        print(
            f"[TRAIN] Training complete."
            f" Epochs: {epoch} | Final reward: {best_reward:.4f} | Elapsed: {elapsed:.1f}s"
        )

        from notifications.emailer import send_email
        send_email(
            subject="[TradingBot] Training Complete",
            body=(
                f"Training complete.\n"
                f"Window:       {train_start} -> {train_end}\n"
                f"Epochs:       {epoch}\n"
                f"Final reward: {best_reward:.4f}\n"
                f"Elapsed:      {elapsed:.1f}s"
            ),
        )

    def _run_epoch(
        self,
        price_data: dict[str, pd.DataFrame],
        indicator_data: dict[str, pd.DataFrame],
    ) -> float:
        """
        Walk forward through every symbol's training bars for one epoch.

        Positions during training: 0=flat, +1=long, -1=short.
        Actions: 0=HOLD (keep current position), 1=BUY (go long), 2=SELL (go short).
        Reward formula: position * pct_return - IMPACT * |position_change|

        Args:
            price_data: Dict of OHLCV DataFrames keyed by symbol.
            indicator_data: Dict of pre-computed indicator DataFrames keyed by symbol.

        Returns:
            Total cumulative reward across all symbols for this epoch.
        """
        total_reward = 0.0

        for sym in self.symbols:
            if sym not in price_data or sym not in indicator_data:
                continue

            closes = price_data[sym]["Close"]
            ind_df = indicator_data[sym]

            # Align closes to the valid indicator index
            closes = closes.reindex(ind_df.index)
            if len(closes) < 2:
                continue

            position = 0  # flat at start of each epoch
            holding_idx = 0

            ind_vals = {col: float(ind_df[col].iloc[0]) for col in ind_df.columns}
            state = self.encoder.encode(ind_vals, holding_idx)
            action = self.learner.querysetstate(state)

            for t in range(len(ind_df) - 1):
                # Long-only mode: action 1 = go long, action 2 = go flat (exit),
                # action 0 = hold current position. Matches live decide() exactly.
                if action == 1:
                    new_position = 1
                elif action == 2:
                    new_position = 0   # exit to flat, not short
                else:
                    new_position = position

                # Reward: position × next-bar return − impact × trade size
                p_t = float(closes.iloc[t])
                p_next = float(closes.iloc[t + 1])
                if p_t == 0:
                    pct_return = 0.0
                else:
                    pct_return = (p_next - p_t) / p_t

                position_change = abs(new_position - position)
                avoided_loss = (1 - new_position) * max(0.0, -pct_return)
                reward = new_position * pct_return + 0.5 * avoided_loss - config.IMPACT * position_change
                total_reward += reward
                position = new_position

                # Long-only: 0 = flat, 1 = long (no short state)
                holding_idx = 1 if position > 0 else 0

                next_ind_vals = {col: float(ind_df[col].iloc[t + 1]) for col in ind_df.columns}
                next_state = self.encoder.encode(next_ind_vals, holding_idx)
                action = self.learner.query(next_state, reward)

        return total_reward

    # ------------------------------------------------------------------
    # Warm start
    # ------------------------------------------------------------------

    def load_pretrained(self) -> bool:
        """
        Attempt to load a saved Q-table and bin edges from QTABLE_PATH.

        On success, the Q-table is restored in the learner and the encoder's
        bin edges are restored so state encoding is consistent with training.
        Sets is_trained=True and returns True. If the file is absent, prints
        a message and returns False so the caller falls back to cold-start training.

        Returns:
            True if the Q-table was loaded successfully, False otherwise.
        """
        try:
            bin_edges, column_order = self.learner.load(config.QTABLE_PATH)
            if bin_edges is not None:
                self.encoder.restore(bin_edges, column_order)
            else:
                logger.warning(
                    "load_pretrained: no bin edges in saved file — "
                    "encoder not restored; consider retraining"
                )
            self.is_trained = True
            print(f"[STARTUP] Warm start — loaded Q-table from {config.QTABLE_PATH}")
            return True
        except FileNotFoundError:
            print(
                f"[TRAIN] No saved Q-table found at {config.QTABLE_PATH}"
                " — cold start required"
            )
            return False

    def retrain(self, train_start: str, train_end: str) -> None:
        """
        Reset all learned state and run a fresh training pipeline.

        Deletes the existing Q-table pickle, resets the encoder's bin edges,
        and reinitialises the Q-table before calling train().

        Args:
            train_start: Start date string "YYYY-MM-DD".
            train_end: End date string "YYYY-MM-DD".
        """
        print(f"[RETRAIN] Retraining from scratch on {train_start} -> {train_end}")
        self.encoder.reset()
        self.learner.reset()
        self.is_trained = False

        if os.path.exists(config.QTABLE_PATH):
            try:
                os.remove(config.QTABLE_PATH)
                logger.info("retrain: deleted old Q-table at %s", config.QTABLE_PATH)
            except OSError as exc:
                logger.warning("retrain: could not delete %s — %s", config.QTABLE_PATH, exc)

        self.train(train_start, train_end)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def decide(self, symbol: str, prices: pd.Series) -> tuple[str, int]:
        """
        Produce a deterministic trade decision for one symbol using the trained Q-table.

        Computes the latest indicator values, encodes the current state (indicators
        + holding position), and reads the best action from the Q-table without
        any randomness or Q-table updates.

        Action mapping:
          0 -> HOLD  (keep current position, 0 shares traded)
          1 -> BUY   (buy TRADE_UNIT shares if below MAX_POSITION_SHARES)
          2 -> SELL  (sell all held shares; no short-selling in live mode)

        Args:
            symbol: Ticker symbol to decide for.
            prices: Recent Close price Series used to compute indicators.

        Returns:
            Tuple of (action_str, shares) where action_str ∈ {"BUY","SELL","HOLD"}.
        """
        ind_vals = {
            name: float(predictor.compute(prices).iloc[-1])
            for name, predictor in self.manager.predictors().items()
        }

        held = self.tracker.holdings.get(symbol, 0)
        if held > 0:
            holding_idx = 1
        elif held < 0:
            holding_idx = 2
        else:
            holding_idx = 0

        state = self.encoder.encode(ind_vals, holding_idx)
        action_int = self.learner.best_action(state)

        if action_int == 1:  # BUY
            if held >= config.MAX_POSITION_SHARES:
                return ("HOLD", 0)
            return ("BUY", config.TRADE_UNIT)
        elif action_int == 2:  # SELL
            if held <= 0:
                return ("HOLD", 0)
            return ("SELL", held)
        else:  # HOLD
            return ("HOLD", 0)

    # ------------------------------------------------------------------
    # Live cycle
    # ------------------------------------------------------------------

    def run_cycle(self) -> None:
        """
        Execute one full hourly decision cycle for all configured symbols.

        For each symbol:
          1. Fetch the most recently completed hourly bar.
          2. Retrieve or build recent price history from the cache.
          3. Compute the realised return since the previous cycle.
          4. Record predictor accuracy and update their weights.
          5. Call decide() and execute the result via the portfolio tracker.

        Skips the entire cycle if is_trained is False.
        """
        if not self.is_trained:
            logger.warning("run_cycle: model not trained — skipping cycle")
            return

        cycle_trade_count_before = sum(
            1 for h in self.tracker.history if h["action"] != "HOLD"
        )
        current_prices: dict[str, float] = {}

        for symbol in self.symbols:
            try:
                bar = fetch_latest_bar(symbol, config.LIVE_INTERVAL)
                current_price = bar["close"]
                current_prices[symbol] = current_price

                # Build or extend cache with the new bar
                cached = price_cache.get(symbol, config.LIVE_INTERVAL)
                if cached is None or cached.empty:
                    from data.fetcher import fetch_prices
                    from datetime import timedelta
                    start = (datetime.now(tz=timezone.utc) - timedelta(days=60)).strftime("%Y-%m-%d")
                    end = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
                    cached = fetch_prices(symbol, start, end, config.LIVE_INTERVAL)
                    price_cache.put(symbol, config.LIVE_INTERVAL, cached)

                prices = cached["Close"]

                # Realised return since last cycle
                prev_price = self._prev_prices.get(symbol)
                if prev_price and prev_price != 0:
                    realized_return = (current_price - prev_price) / prev_price
                    self.manager.record_accuracy(prices, realized_return)
                    self.updater.update()
                self._prev_prices[symbol] = current_price

                # Decide and execute
                action, shares = self.decide(symbol, prices)
                signals = self.manager.get_all_signals(prices)
                weighted = self.manager.get_weighted_signal(prices)
                signals_str = " ".join(
                    f"{k.replace('Predictor', '')}:{v:+d}"
                    for k, v in signals.items()
                )
                reason = f"Weighted signal {weighted:+.2f} ({signals_str})"
                self.tracker.execute(symbol, action, shares, current_price, reason)

            except Exception as exc:
                logger.error("run_cycle: error processing %s — %s", symbol, exc, exc_info=True)

        total = self.tracker.portfolio_value(current_prices)
        ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        print(f"[CYCLE] {ts} ET | Portfolio: ${total:,.2f}")

        # Build cycle summary for email
        new_trades = [
            h for h in self.tracker.history
            if h["action"] != "HOLD"
        ][cycle_trade_count_before:]
        weights = self.manager.get_weights()
        weights_str = "  ".join(
            f"{k.replace('Predictor', '')}={v:.3f}" for k, v in weights.items()
        )
        trades_str = (
            "\n".join(
                f"  {t['symbol']} {t['action']} {t['shares']} @ ${t['price']:.2f}"
                for t in new_trades
            )
            if new_trades else "  (no trades this cycle)"
        )
        from notifications.emailer import send_email
        send_email(
            subject=f"[TradingBot] Cycle {ts}",
            body=(
                f"Cycle complete: {ts}\n"
                f"Portfolio: ${total:,.2f}\n\n"
                f"Trades:\n{trades_str}\n\n"
                f"Weights: {weights_str}"
            ),
        )

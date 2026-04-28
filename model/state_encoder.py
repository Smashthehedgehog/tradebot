import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class StateEncoder:
    """
    Converts continuous indicator values and a holding position into a single
    discrete integer state index suitable for Q-table lookup.

    Bin edges are computed once from training data (fit()) and reused for all
    subsequent encoding, including live bars, so the mapping is always
    consistent with what the Q-table learned.
    """

    def __init__(self, num_bins: int) -> None:
        """
        Initialise the encoder with the desired number of quantile bins.

        Args:
            num_bins: Number of equal-frequency bins per indicator, e.g. 8.
                      Total states = 3 holdings × num_bins^num_indicators.
        """
        self.num_bins = num_bins
        self.bin_edges: list[np.ndarray] | None = None
        self._fitted: bool = False
        self._column_order: list[str] = []

    def fit(self, indicator_df: pd.DataFrame) -> None:
        """
        Compute equal-frequency bin edges for each indicator column using
        training data only.

        Bin edges are derived from quantiles so each bin contains roughly the
        same number of observations, giving the Q-table balanced coverage of
        the state space. Must be called exactly once before encode(); raises
        RuntimeError if called a second time without reset().

        Args:
            indicator_df: DataFrame where each column is one indicator's full
                          training time series. NaN rows are dropped per column
                          before computing quantiles.

        Raises:
            RuntimeError: If fit() is called again before reset().
        """
        if self._fitted:
            raise RuntimeError(
                "StateEncoder is already fitted. Call reset() before fitting again."
            )

        quantile_points = np.linspace(0.0, 1.0, self.num_bins + 1)
        self.bin_edges = []
        self._column_order = list(indicator_df.columns)

        for col in self._column_order:
            values = indicator_df[col].dropna().values
            edges = np.quantile(values, quantile_points)
            self.bin_edges.append(edges)
            logger.debug("fit: column='%s' edges=%s", col, edges)

        self._fitted = True

    def reset(self) -> None:
        """
        Clear bin edges so fit() can be called again.

        Used by TradingEngine.retrain() to prepare for a fresh training run
        without instantiating a new StateEncoder object.
        """
        self.bin_edges = None
        self._fitted = False
        self._column_order = []

    def restore(self, bin_edges: list[np.ndarray], column_order: list[str] | None = None) -> None:
        """
        Directly load previously saved bin edges, skipping the fit step.

        Used during warm-start loading so the encoder is consistent with the
        saved Q-table without re-fetching or re-processing training data.

        Args:
            bin_edges: List of np.ndarray edge arrays, one per indicator column,
                       as returned by learner.load().
            column_order: Optional list of column names in the same order as
                          bin_edges. Defaults to generic positional names if None.
        """
        self.bin_edges = bin_edges
        self._fitted = True
        if column_order is not None:
            self._column_order = column_order
        else:
            self._column_order = [f"indicator_{i}" for i in range(len(bin_edges))]

    def encode(self, indicator_values: dict[str, float], holding: int) -> int:
        """
        Convert current indicator values and holding position to a state integer.

        Each indicator value is placed into one of num_bins bins using the
        stored edges. The bin indices are combined into a single integer using
        mixed-radix encoding, then offset by the holding index so flat, long,
        and short positions map to distinct regions of the state space.

        Formula:
            state = holding * (num_bins ^ num_indicators) + bin_combo

        Args:
            indicator_values: Dict mapping each indicator name to its current
                              float value, in the same order as fitted columns.
            holding: 0=flat, 1=long, 2=short.

        Returns:
            Integer state index in [0, num_states).

        Raises:
            RuntimeError: If fit() or restore() has not been called yet.
        """
        if self.bin_edges is None:
            raise RuntimeError(
                "StateEncoder not fitted. Call fit() or restore() before encode()."
            )

        bin_indices: list[int] = []
        for i, col in enumerate(self._column_order):
            val = indicator_values.get(col, 0.0)
            edges = self.bin_edges[i]
            idx = int(np.searchsorted(edges, val, side="right")) - 1
            idx = max(0, min(idx, self.num_bins - 1))
            bin_indices.append(idx)

        n = len(bin_indices)
        bin_combo = 0
        for i, b in enumerate(bin_indices):
            bin_combo += b * (self.num_bins ** (n - 1 - i))

        state = holding * (self.num_bins ** n) + bin_combo
        return int(state)

    def num_states(self, num_indicators: int) -> int:
        """
        Return the total number of discrete states for a given indicator count.

        Used when constructing the QLearner so the Q-table is sized correctly.

        Args:
            num_indicators: Number of predictor signals used as features.

        Returns:
            3 * num_bins^num_indicators (three holding positions × indicator space).
        """
        return 3 * (self.num_bins ** num_indicators)

import logging
from collections import OrderedDict

import pandas as pd

import config
from indicators.base import BasePredictor

logger = logging.getLogger(__name__)


class PredictorManager:
    """
    Central registry that holds all active predictor instances and their weights.

    Weights always sum to 1.0 and are kept above MIN_PREDICTOR_WEIGHT so no
    predictor is ever completely silenced. Predictors can be added at runtime
    via register() without touching any other part of the codebase.
    """

    def __init__(self, predictors: list[BasePredictor]) -> None:
        """
        Initialise the registry with equal weights across all predictors.

        Args:
            predictors: List of BasePredictor instances to register.
        """
        self._predictors: OrderedDict[str, BasePredictor] = OrderedDict()
        for p in predictors:
            self._predictors[p.name()] = p
        self._equalize_weights()

    # ------------------------------------------------------------------
    # Signal helpers
    # ------------------------------------------------------------------

    def get_weighted_signal(self, prices: pd.Series) -> float:
        """
        Compute the weighted sum of all predictor votes for the latest bar.

        Each predictor's vote (+1, -1, or 0) is multiplied by its current
        weight; the results are summed to produce a value in [-1.0, +1.0].

        Args:
            prices: Close price Series used by each predictor's compute().

        Returns:
            Weighted signal float in the range [-1.0, +1.0].
        """
        total = 0.0
        for p in self._predictors.values():
            val = float(p.compute(prices).iloc[-1])
            total += p.weight * p.signal(val)
        return total

    def get_all_signals(self, prices: pd.Series) -> dict[str, int]:
        """
        Return each predictor's raw vote for the latest bar, keyed by name.

        Used for logging and building the reason string in trade output.

        Args:
            prices: Close price Series.

        Returns:
            Dict mapping predictor name to its integer vote (+1, -1, or 0).
        """
        return {
            name: p.signal(float(p.compute(prices).iloc[-1]))
            for name, p in self._predictors.items()
        }

    def record_accuracy(self, prices: pd.Series, realized_return: float) -> None:
        """
        Append a score to each predictor's accuracy_history based on whether its
        vote matched the direction of the realised return.

        Neutral votes (0) score 0.5 — treated as abstentions, neither rewarded nor
        penalised. Recording 0.0 for neutral would push all predictors to always take
        a stance, causing overtrading during the sideways conditions that dominate
        hourly markets 60–70% of the time.

        Args:
            prices: Close price Series used to compute each predictor's vote.
            realized_return: Actual percent return since the previous cycle.
        """
        for p in self._predictors.values():
            val = float(p.compute(prices).iloc[-1])
            vote = p.signal(val)
            if vote == 0:
                correct = 0.5  # abstention — no information, no penalty
            else:
                correct = 1.0 if (vote > 0) == (realized_return > 0) else 0.0
            p.record_accuracy(correct)

    # ------------------------------------------------------------------
    # Weight management
    # ------------------------------------------------------------------

    def get_weights(self) -> dict[str, float]:
        """
        Return the current weight of every registered predictor.

        Returns:
            Dict mapping predictor name to its float weight.
        """
        return {name: p.weight for name, p in self._predictors.items()}

    def set_weight(self, name: str, weight: float) -> None:
        """
        Override one predictor's weight, then re-normalise all weights so
        they sum to 1.0 and no weight falls below MIN_PREDICTOR_WEIGHT.

        Args:
            name: Name of the predictor to update, e.g. "SMARatioPredictor".
            weight: Desired raw weight before normalisation.
        """
        if name not in self._predictors:
            logger.warning("set_weight: unknown predictor '%s'", name)
            return
        self._predictors[name].weight = max(weight, config.MIN_PREDICTOR_WEIGHT)
        self._normalize_weights()

    def set_weights_bulk(self, weights: dict[str, float]) -> None:
        """
        Set raw weights for multiple predictors at once, then normalise once.

        Used by WeightUpdater to avoid repeated normalisation on each predictor.

        Args:
            weights: Dict mapping predictor name to its new raw weight value.
        """
        for name, w in weights.items():
            if name in self._predictors:
                self._predictors[name].weight = max(w, config.MIN_PREDICTOR_WEIGHT)
        self._normalize_weights()

    def register(self, predictor: BasePredictor) -> None:
        """
        Add a new predictor to the registry at runtime and redistribute weights
        equally across all predictors including the new one.

        This is the primary extension point — adding a news or earnings predictor
        requires only calling this method and retraining.

        Args:
            predictor: A new BasePredictor instance to add.
        """
        self._predictors[predictor.name()] = predictor
        self._equalize_weights()
        logger.info("register: added predictor '%s'", predictor.name())

    def predictors(self) -> OrderedDict:
        """
        Return the internal OrderedDict of all registered predictors.

        Returns:
            OrderedDict mapping name to BasePredictor instance.
        """
        return self._predictors

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _equalize_weights(self) -> None:
        """Set all weights to 1/N where N is the number of predictors."""
        n = len(self._predictors)
        if n == 0:
            return
        w = 1.0 / n
        for p in self._predictors.values():
            p.weight = w

    def _normalize_weights(self) -> None:
        """
        Normalise weights so they sum to 1.0, re-applying the MIN_PREDICTOR_WEIGHT
        floor after normalisation to prevent any weight from reaching zero.
        """
        total = sum(p.weight for p in self._predictors.values())
        if total == 0:
            self._equalize_weights()
            return
        for p in self._predictors.values():
            p.weight = p.weight / total

        # Apply floor and re-normalise a second time if clamping shifted the sum
        for p in self._predictors.values():
            p.weight = max(p.weight, config.MIN_PREDICTOR_WEIGHT)
        total = sum(p.weight for p in self._predictors.values())
        for p in self._predictors.values():
            p.weight /= total

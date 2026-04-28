import logging

import config
from model.predictor_manager import PredictorManager

logger = logging.getLogger(__name__)


class WeightUpdater:
    """
    Automatically adjusts predictor weights after every trading step based on
    each predictor's recent accuracy, using an exponential moving average so
    recent performance matters more than older history.
    """

    def __init__(self, manager: PredictorManager) -> None:
        """
        Initialise the updater with equal EMA accuracy for all predictors.

        Args:
            manager: The PredictorManager whose weights will be adjusted.
        """
        self._manager = manager
        self.step_count: int = 0
        self._ema_accuracies: dict[str, float] = {
            name: 0.5 for name in manager.get_weights()
        }

    def update(self) -> None:
        """
        Recompute EMA accuracy for each predictor, derive new weights, and
        write them back to the manager.

        Called after every completed trading step (hourly in live mode, once
        per epoch during training). The EMA formula is:
            ema = WEIGHT_DECAY * prev_ema + (1 - WEIGHT_DECAY) * latest_accuracy

        Steps:
          1. Pull the most recent accuracy entry from each predictor's history.
          2. Update that predictor's EMA accuracy.
          3. Push all updated EMA values as raw weights to the manager via
             set_weights_bulk(), which normalises and clamps them in one pass.
          4. Increment step_count.
        """
        new_raw: dict[str, float] = {}

        for name, predictor in self._manager.predictors().items():
            if not predictor.accuracy_history:
                new_raw[name] = self._ema_accuracies.get(name, 0.5)
                continue

            latest = predictor.accuracy_history[-1]
            prev_ema = self._ema_accuracies.get(name, 0.5)
            ema = config.WEIGHT_DECAY * prev_ema + (1.0 - config.WEIGHT_DECAY) * latest
            self._ema_accuracies[name] = ema
            new_raw[name] = ema

        self._manager.set_weights_bulk(new_raw)
        self.step_count += 1
        logger.debug(
            "weight_updater.update step=%d ema_acc=%s weights=%s",
            self.step_count,
            {k: round(v, 4) for k, v in self._ema_accuracies.items()},
            {k: round(v, 4) for k, v in self._manager.get_weights().items()},
        )

    def get_ema_accuracies(self) -> dict[str, float]:
        """
        Return the current exponential moving average accuracy for each predictor.

        Used by the API's GET /weights route to expose self-tuning state to the
        frontend without exposing internal objects.

        Returns:
            Dict mapping predictor name to its current EMA accuracy float.
        """
        return dict(self._ema_accuracies)

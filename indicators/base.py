import logging
from abc import ABC, abstractmethod

import pandas as pd

logger = logging.getLogger(__name__)

_MAX_ACCURACY_HISTORY = 500


class BasePredictor(ABC):
    """
    Abstract base class that every predictor/indicator must subclass.

    Enforces a common interface so the PredictorManager can treat all
    predictors uniformly regardless of their internal logic.
    """

    def __init__(self) -> None:
        """
        Initialise default weight and an empty accuracy history.
        """
        self.weight: float = 1.0
        self.accuracy_history: list[float] = []

    @abstractmethod
    def compute(self, prices: pd.Series) -> pd.Series:
        """
        Compute a continuous signal value for every bar in the price series.

        Implementations must not introduce lookahead — only data at or before
        each timestamp may be used when producing that timestamp's value.

        Args:
            prices: Close price Series with a DatetimeIndex.

        Returns:
            Same-length Series of continuous float signal values.
        """

    @abstractmethod
    def signal(self, value: float) -> int:
        """
        Map a single continuous indicator value to a discrete trade vote.

        Args:
            value: A single float produced by compute().

        Returns:
            +1 (bullish), -1 (bearish), or 0 (neutral).
        """

    def name(self) -> str:
        """
        Return the class name, used as the registry key in PredictorManager.

        Returns:
            String class name, e.g. "SMARatioPredictor".
        """
        return self.__class__.__name__

    def votes(self, prices: pd.Series) -> pd.Series:
        """
        Compute the full vote Series by running compute() then mapping each
        value through signal().

        Args:
            prices: Close price Series with a DatetimeIndex.

        Returns:
            Integer Series of +1, -1, or 0 values, same length as prices.
        """
        return self.compute(prices).map(self.signal)

    def record_accuracy(self, correct: float) -> None:
        """
        Append a 1.0 (correct) or 0.0 (incorrect) to accuracy_history,
        capping the list at _MAX_ACCURACY_HISTORY entries (FIFO).

        Args:
            correct: 1.0 if this predictor's last vote matched the realised
                     return direction, 0.0 otherwise.
        """
        self.accuracy_history.append(correct)
        if len(self.accuracy_history) > _MAX_ACCURACY_HISTORY:
            self.accuracy_history.pop(0)

import logging
import os
import pickle

import numpy as np

logger = logging.getLogger(__name__)


class QLearner:
    """
    Tabular Q-learning engine implemented in pure numpy.

    Maintains a Q-table of shape (num_states, num_actions) and updates it via
    the Bellman equation after every observed transition. Supports both
    exploratory training (ε-greedy) and deterministic inference (best_action).
    """

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        alpha: float,
        gamma: float,
        rar: float,
        radr: float,
    ) -> None:
        """
        Initialise the Q-table with near-zero random noise to break symmetry.

        Args:
            num_states: Total number of discrete states (rows in the Q-table).
            num_actions: Number of possible actions (columns in the Q-table).
            alpha: Learning rate in (0, 1]; controls how much each update moves
                   the Q-value toward the new estimate.
            gamma: Discount factor in [0, 1]; weights future rewards vs immediate.
            rar: Initial random action rate in [0, 1]; probability of choosing
                 a random action instead of the greedy one.
            radr: Random action decay rate applied each step as rar *= radr;
                  drives exploration toward exploitation over time.
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self._initial_rar = rar

        self.Q = np.zeros((num_states, num_actions))
        self.Q += np.random.uniform(-0.001, 0.001, self.Q.shape)

        self.s: int = 0
        self.a: int = 0

    def querysetstate(self, s: int) -> int:
        """
        Set the current state and select an action without updating the Q-table.

        Used at the start of each training epoch to initialise the state before
        the first Bellman update. Action selection is ε-greedy.

        Args:
            s: Integer state index in [0, num_states).

        Returns:
            Selected action integer in [0, num_actions).
        """
        self.s = s
        self.a = self._select_action(s)
        return self.a

    def query(self, s_prime: int, r: float) -> int:
        """
        Apply a Bellman update for the previous (s, a) transition, then select
        the next action from s_prime using ε-greedy and decay rar.

        Bellman update:
            Q[s, a] += alpha * (r + gamma * max(Q[s']) - Q[s, a])

        Args:
            s_prime: Next state index observed after taking action self.a.
            r: Reward received for the transition (self.s, self.a) → s_prime.

        Returns:
            Next action selected from s_prime.
        """
        self.Q[self.s, self.a] += self.alpha * (
            r + self.gamma * np.max(self.Q[s_prime]) - self.Q[self.s, self.a]
        )
        self.rar = max(self.rar * self.radr, 0.0)
        self.s = s_prime
        self.a = self._select_action(s_prime)
        return self.a

    def best_action(self, s: int) -> int:
        """
        Return the greedy action for state s with NO randomness and NO Q-update.

        Used exclusively during inference (decide()) so live decisions are
        always deterministic and reproducible.

        Args:
            s: Integer state index in [0, num_states).

        Returns:
            Action index with the highest Q-value for state s.
        """
        return int(np.argmax(self.Q[s]))

    def save(self, path: str, bin_edges: list | None = None, column_order: list | None = None) -> None:
        """
        Pickle the Q-table, bin edges, and column order to disk.

        Storing bin edges and column order alongside the Q-table ensures that
        warm-start loading is always self-consistent — the encoder can be
        restored without re-fetching training data.

        Args:
            path: File path to write, e.g. "logs/qtable.pkl".
            bin_edges: StateEncoder bin edges to bundle into the pickle.
            column_order: List of indicator column names matching bin_edges order.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "qtable": self.Q,
            "bin_edges": bin_edges,
            "column_order": column_order,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        logger.info("q_learner.save: Q-table saved to %s", path)

    def load(self, path: str) -> tuple[list | None, list | None]:
        """
        Load a previously saved Q-table from disk and restore self.Q.

        Returns the saved bin_edges and column_order so the caller can restore
        the StateEncoder without re-fitting on training data.

        Args:
            path: File path to read, e.g. "logs/qtable.pkl".

        Returns:
            Tuple of (bin_edges, column_order); either may be None if the file
            was saved without them (legacy format).

        Raises:
            FileNotFoundError: If the file does not exist at path.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No saved Q-table found at '{path}'. "
                "Run with --retrain to train from scratch."
            )
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self.Q = payload["qtable"]
        logger.info("q_learner.load: Q-table loaded from %s", path)
        return payload.get("bin_edges"), payload.get("column_order")

    def reset(self) -> None:
        """
        Re-initialise the Q-table and restore rar to its original value.

        Called by TradingEngine.retrain() before a fresh training run so the
        learner starts from scratch without needing to be reconstructed.
        """
        self.Q = np.zeros((self.num_states, self.num_actions))
        self.Q += np.random.uniform(-0.001, 0.001, self.Q.shape)
        self.rar = self._initial_rar
        self.s = 0
        self.a = 0
        logger.info("q_learner.reset: Q-table re-initialised")

    def _select_action(self, s: int) -> int:
        """
        ε-greedy action selection: random with probability rar, greedy otherwise.

        Args:
            s: State index to select an action for.

        Returns:
            Action integer in [0, num_actions).
        """
        if np.random.random() < self.rar:
            return int(np.random.randint(self.num_actions))
        return int(np.argmax(self.Q[s]))

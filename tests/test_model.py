import numpy as np
import pandas as pd
import pytest

import config
from indicators.technical import (
    BollingerPercentBPredictor,
    MACDHistogramPredictor,
    SMARatioPredictor,
)
from model.predictor_manager import PredictorManager
from model.q_learner import QLearner
from model.state_encoder import StateEncoder
from model.weight_updater import WeightUpdater


def _make_manager() -> PredictorManager:
    return PredictorManager([
        SMARatioPredictor(),
        BollingerPercentBPredictor(),
        MACDHistogramPredictor(),
    ])


def test_state_encoder_fit_encode():
    encoder = StateEncoder(num_bins=4)
    rng = np.random.default_rng(0)
    n = 200
    ind_df = pd.DataFrame({
        "sma_ratio": rng.uniform(0.8, 1.2, n),
        "bollinger": rng.uniform(0.0, 1.0, n),
        "macd": rng.uniform(-0.5, 0.5, n),
    })
    encoder.fit(ind_df)

    state = encoder.encode({"sma_ratio": 1.0, "bollinger": 0.5, "macd": 0.0}, holding=0)

    assert isinstance(state, int)
    assert 0 <= state < encoder.num_states(3)


def test_qlearner_query_updates_qtable():
    learner = QLearner(
        num_states=10, num_actions=3,
        alpha=0.2, gamma=0.9,
        rar=0.0, radr=1.0,  # rar=0 forces greedy so action is deterministic
    )
    s0 = 0
    a0 = learner.querysetstate(s0)
    q_before = learner.Q[s0, a0]

    learner.query(s_prime=1, r=1.0)

    assert learner.Q[s0, a0] != q_before


def test_qlearner_best_action_deterministic():
    learner = QLearner(
        num_states=10, num_actions=3,
        alpha=0.2, gamma=0.9,
        rar=0.5, radr=1.0,
    )
    s = 5
    assert learner.best_action(s) == learner.best_action(s)


def test_predictor_manager_weights_sum_to_one():
    manager = _make_manager()

    weights = manager.get_weights()
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-9)

    manager.set_weight("SMARatioPredictor", 0.6)
    weights = manager.get_weights()
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-9)


def test_weight_updater_changes_weights():
    manager = _make_manager()
    updater = WeightUpdater(manager)

    initial_weights = dict(manager.get_weights())

    # Inject different accuracy values per predictor so the EMA diverges
    names = list(manager.predictors().keys())
    for i, name in enumerate(names):
        manager.predictors()[name].accuracy_history.append(1.0 if i == 0 else 0.0)

    updater.update()

    updated_weights = manager.get_weights()

    # Weights must still sum to 1.0
    assert sum(updated_weights.values()) == pytest.approx(1.0, abs=1e-9)
    # At least one weight must have moved from the initial equal value
    assert any(
        abs(updated_weights[k] - initial_weights[k]) > 1e-9
        for k in updated_weights
    )

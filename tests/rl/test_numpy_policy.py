"""The torch-free :class:`NumpyPolicy` must match the torch net it replaces."""
import numpy as np
import pytest

from othello_rl.environment.board import Board
from othello_rl.environment.environment import encode_observation, legal_action_mask
from othello_rl.rl.agent import DQNAgent, NetworkConfig
from othello_rl.rl.numpy_policy import NumpyPolicy


def _sd_to_arrays(agent: DQNAgent) -> dict:
    return {k: v.detach().cpu().numpy().astype(np.float32)
            for k, v in agent.net.state_dict().items()}


@pytest.fixture(scope="module")
def pair():
    agent = DQNAgent(NetworkConfig(channels=16, blocks=2, hidden=32), seed=0)
    npol = NumpyPolicy(_sd_to_arrays(agent),
                       {"channels": 16, "blocks": 2, "hidden": 32})
    return agent, npol


def _positions(n=60, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        b = Board.initial()
        for _ in range(int(rng.integers(0, 34))):
            legal = b.legal_moves()
            if not legal:
                b = b.apply(None)
                continue
            b = b.apply(legal[int(rng.integers(0, len(legal)))])
        if not b.is_terminal() and b.legal_moves():
            out.append(b)
    return out


def test_q_values_match_torch(pair):
    agent, npol = pair
    worst = 0.0
    for b in _positions():
        obs, msk = encode_observation(b), legal_action_mask(b)
        qt = agent.q_values(obs, msk)
        qn = npol.q_values(obs, msk)
        finite = np.isfinite(qt) & np.isfinite(qn)
        worst = max(worst, float(np.abs(qt[finite] - qn[finite]).max()))
    assert worst < 1e-4, worst


def test_greedy_and_select_match_torch(pair):
    agent, npol = pair
    for b in _positions(seed=1):
        obs, msk = encode_observation(b), legal_action_mask(b)
        assert agent.greedy_act(obs, msk) == npol.greedy_act(obs, msk)
        assert agent.select_move(b) == npol.select_move(b)


def test_othellobot_loads_from_npz(tmp_path):
    from othello_rl.webapp.bot_service import OthelloBot
    agent = DQNAgent(NetworkConfig(channels=16, blocks=2, hidden=32), seed=2)
    npz = tmp_path / "policy.npz"
    import json
    arrays = _sd_to_arrays(agent)
    arrays["net_config_json"] = np.array(json.dumps({"channels": 16, "blocks": 2, "hidden": 32}))
    np.savez_compressed(npz, **arrays)

    from othello_rl.webapp.moves import parse_game
    line = parse_game("c4c3d3")
    bot = OthelloBot.load(str(npz))
    b = Board.initial()
    assert 0 <= bot.select_action(b) < 64
    assert bot.best_move(b)["action"] in [r * 8 + c for r, c in b.legal_moves()]
    d = bot.analyse_line(line)
    assert len(d["plies"]) == len(line)

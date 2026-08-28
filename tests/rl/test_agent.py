import numpy as np
import torch

from othello_rl.agents import RandomAgent
from othello_rl.environment.board import Board
from othello_rl.environment.environment import encode_observation, legal_action_mask
from othello_rl.evaluation.tournament import play_game
from othello_rl.rl.agent import DQNAgent, NetworkConfig


def _small_agent(seed=0):
    return DQNAgent(NetworkConfig(channels=16, blocks=2, hidden=32), seed=seed)


def test_greedy_act_returns_legal_action():
    agent = _small_agent()
    st = Board.initial()
    for _ in range(6):
        obs = encode_observation(st)
        mask = legal_action_mask(st)
        a = agent.greedy_act(obs, mask)
        assert mask[a]
        st = st.apply(None if a == 64 else divmod(a, 8))


def test_epsilon_exploration_only_picks_legal():
    agent = _small_agent(seed=1)
    st = Board.initial()
    seen = set()
    for _ in range(200):
        mask = legal_action_mask(st)
        a = agent.act(encode_observation(st), mask, epsilon=1.0)
        assert mask[a]
        seen.add(a)
    assert len(seen) > 1  # exploration actually varies


def test_select_move_is_agent_interface_and_legal():
    agent = _small_agent()
    result = play_game(agent, RandomAgent(seed=2), seed=0)
    assert result.plies > 0
    assert result.black_score + result.white_score <= 64


def test_checkpoint_roundtrip(tmp_path):
    agent = _small_agent(seed=3)
    agent.meta.train_steps = 4242
    agent.meta.episodes = 7
    # perturb weights so we're not comparing a fresh net to a fresh net
    with torch.no_grad():
        for p in agent.net.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    path = tmp_path / "ckpt.pt"
    agent.save(path)

    loaded = DQNAgent.from_checkpoint(path)
    assert loaded.meta.train_steps == 4242
    assert loaded.meta.episodes == 7
    for p1, p2 in zip(agent.net.parameters(), loaded.net.parameters()):
        assert torch.allclose(p1, p2)

    # identical greedy decisions on the same inputs
    st = Board.initial().apply((2, 3)).apply((2, 2))
    obs, mask = encode_observation(st), legal_action_mask(st)
    assert agent.greedy_act(obs, mask) == loaded.greedy_act(obs, mask)


def test_clone_network_is_detached_copy():
    agent = _small_agent()
    twin = agent.clone_network()
    for p in twin.parameters():
        assert not p.requires_grad
    agent.net.eval()
    x = torch.randn(4, 3, 8, 8)
    with torch.no_grad():
        assert torch.allclose(agent.net(x), twin(x))

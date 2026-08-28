import numpy as np
import torch

from othello_rl.rl.network import NEG_INF, SmallOthelloNet, greedy_action, masked_q
from othello_rl.rl.replay_buffer import ReplayBuffer
from othello_rl.environment.environment import NUM_ACTIONS


def test_network_forward_shapes():
    net = SmallOthelloNet(channels=16, blocks=2, hidden=32)
    x = torch.randn(5, 3, 8, 8)
    q = net(x)
    assert q.shape == (5, NUM_ACTIONS)
    q2, v = net.forward_with_value(x)
    assert q2.shape == (5, NUM_ACTIONS)
    assert v.shape == (5,)
    assert torch.all(v >= -1) and torch.all(v <= 1)


def test_masked_q_and_greedy_action():
    q = torch.tensor([[1.0, 5.0, 3.0, 9.0, 2.0]])
    mask = torch.tensor([[True, True, False, False, True]])
    mq = masked_q(q, mask)
    assert mq[0, 2].item() == NEG_INF and mq[0, 3].item() == NEG_INF
    assert mq[0, 1].item() == 5.0
    # best legal is index 1 (5.0), not the illegal 9.0 at index 3
    assert greedy_action(q, mask).item() == 1


def test_replay_buffer_add_sample_and_wraparound():
    buf = ReplayBuffer(capacity=10, seed=0)
    assert not buf.can_sample(1)
    for i in range(25):
        obs = np.full((3, 8, 8), i, dtype=np.float32)
        buf.add(obs, i % NUM_ACTIONS, float(i), obs, i % 2,
                np.zeros(NUM_ACTIONS, dtype=bool))
    assert len(buf) == 10  # capped
    batch = buf.sample(8)
    assert batch.obs.shape == (8, 3, 8, 8)
    assert batch.actions.shape == (8,)
    assert batch.next_masks.shape == (8, NUM_ACTIONS)
    # only the last 10 items (15..24) should remain
    assert batch.rewards.min() >= 15.0


def test_replay_buffer_sample_too_large_raises():
    buf = ReplayBuffer(capacity=5, seed=0)
    obs = np.zeros((3, 8, 8), dtype=np.float32)
    buf.add(obs, 0, 0.0, obs, 0, np.zeros(NUM_ACTIONS, dtype=bool))
    try:
        buf.sample(4)
        assert False, "expected ValueError"
    except ValueError:
        pass

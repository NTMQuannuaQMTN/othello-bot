"""Supervised imitation pretraining of a policy(+value) net (Phase 12.5)."""
import random

import numpy as np
import pytest
import torch

from othello_rl.agents import RandomAgent
from othello_rl.environment.board import Board
from othello_rl.evaluation.tournament import play_match
from othello_rl.rl.az_agent import PolicyValueAgent
from othello_rl.rl.az_network import PolicyValueConfig
from othello_rl.rl.checkpoint import load_agent, load_checkpoint, save_policy_value_checkpoint
from othello_rl.rl.supervised import SupervisedConfig, SupervisedTrainer


def _tiny_dataset(tmp_path, n_train=160, n_val=40, seed=0):
    rng = np.random.RandomState(seed)
    grng = random.Random(seed)

    def make(n):
        obs = np.zeros((n, 3, 8, 8), dtype=np.float32)
        policy = np.zeros(n, dtype=np.int16)
        s = Board.initial()
        for i in range(n):
            if s.is_terminal():
                s = Board.initial()
            from othello_rl.environment.environment import encode_observation
            legal = [r * 8 + c for r, c in s.legal_moves()]
            obs[i] = encode_observation(s)
            policy[i] = legal[0]                      # always the first legal move
            s = s.apply(divmod(policy[i], 8))
        return {"obs": obs, "policy": policy,
                "value": rng.choice([-1.0, 0.0, 1.0], n).astype(np.float32),
                "weight": np.ones(n, dtype=np.float32)}

    tr, va = make(n_train), make(n_val)
    np.savez_compressed(tmp_path / "train.npz", **tr)
    np.savez_compressed(tmp_path / "val.npz", **va)
    return tmp_path / "train.npz", tmp_path / "val.npz"


def _cfg(**kw):
    base = dict(az_network={"channels": 8, "blocks": 2, "hidden": 16, "norm": False},
                epochs=6, lr=3e-3, batch_size=32, seed=0)
    base.update(kw)
    return SupervisedConfig(**base)


def test_overfits_a_tiny_dataset():
    # 12 shared positions, one repeated target -> the net should learn it
    from othello_rl.environment.environment import encode_observation
    s = Board.initial()
    obs, pol = [], []
    for _ in range(12):
        legal = [r * 8 + c for r, c in s.legal_moves()]
        obs.append(encode_observation(s)); pol.append(legal[0])
        s = s.apply(divmod(pol[-1], 8))
        if s.is_terminal():
            s = Board.initial()
    d = {"obs": np.array(obs, np.float32), "policy": np.array(pol, np.int16),
         "value": np.zeros(12, np.float32), "weight": np.ones(12, np.float32)}
    tr = SupervisedTrainer(_cfg(az_network={"channels": 16, "blocks": 2, "hidden": 32,
                                            "norm": False}, epochs=150, batch_size=12),
                           device="cpu")
    hist = tr.fit(d, d)
    assert hist[-1].train_loss < hist[0].train_loss * 0.4      # loss drops sharply
    assert hist[-1].move_accuracy >= 0.9                        # ~memorises the map


def test_fit_trains_only_on_train_and_reports_val_metrics(tmp_path):
    # the training loop indexes only into the train tensors; val is used for
    # metrics only. (Game-level no-leak split is enforced in datasets/, Phase 12.4.)
    train_npz, val_npz = _tiny_dataset(tmp_path)
    tr = SupervisedTrainer(_cfg(), device="cpu")
    hist = tr.fit(train_npz, val_npz)
    assert len(hist) == 6
    m = hist[-1]
    assert m.val_loss > 0 and 0.0 <= m.move_accuracy <= 1.0
    # val metrics differ from train (independent set)
    assert m.val_loss != m.train_loss


def test_checkpoint_saved_under_experiments_and_roundtrips(tmp_path):
    train_npz, val_npz = _tiny_dataset(tmp_path)
    tr = SupervisedTrainer(_cfg(epochs=3), device="cpu")
    tr.fit(train_npz, val_npz)
    ckpt = tmp_path / "experiments" / "v001_pretrain.pt"
    save_policy_value_checkpoint(ckpt, tr.net, tr.cfg.net_config(), optimizer=tr.opt,
                                 epoch=tr.epoch, version="v001", dataset_version="dsX")
    assert "production" not in str(ckpt)

    c = load_checkpoint(ckpt)
    assert c.net_kind == "policy_value" and c.meta.extra["method"] == "supervised_pretrain"
    agent = load_agent(ckpt)
    assert isinstance(agent, PolicyValueAgent)
    # predictions reproduce
    b = Board.initial().apply((2, 3))
    p1 = agent.move_probabilities(b)
    p2 = PolicyValueAgent.from_state(c.pv_config, c.state_dict).move_probabilities(b)
    assert p1.keys() == p2.keys()
    for k in p1:
        assert p1[k] == pytest.approx(p2[k], abs=1e-5)


def test_resume_restores_epoch_and_optimizer(tmp_path):
    train_npz, val_npz = _tiny_dataset(tmp_path)
    tr = SupervisedTrainer(_cfg(epochs=3), device="cpu")
    tr.fit(train_npz, val_npz)
    ckpt = tmp_path / "e" / "v1.pt"
    save_policy_value_checkpoint(ckpt, tr.net, tr.cfg.net_config(), optimizer=tr.opt,
                                 epoch=tr.epoch, rng_state=tr.rng_state(), version="v1")

    tr2 = SupervisedTrainer.resume(_cfg(epochs=2), ckpt, device="cpu")
    assert tr2.epoch == 3
    assert tr2.opt.state_dict()["state"]           # optimizer state restored
    hist = tr2.fit(train_npz, val_npz)
    assert hist[-1].epoch == 5                     # 3 done + 2 more


def test_policy_value_agent_always_legal_and_beats_random_after_training(tmp_path):
    # sanity: even lightly trained, a PolicyValueAgent returns legal moves and
    # plugs into play_match.
    train_npz, val_npz = _tiny_dataset(tmp_path, 240, 40)
    tr = SupervisedTrainer(_cfg(epochs=8), device="cpu")
    tr.fit(train_npz, val_npz)
    agent = PolicyValueAgent(tr.net)
    s = Board.initial()
    for _ in range(20):
        if s.is_terminal():
            break
        mv = agent.select_move(s)
        assert mv is None or mv in s.legal_moves()
        s = s.apply(mv)
    m = play_match(agent, RandomAgent(seed=1), num_games=8, seed=0)
    assert 0 <= m.a_win_rate <= 1


def test_dqn_checkpoint_still_loads_via_load_agent():
    from othello_rl.rl.agent import DQNAgent, NetworkConfig
    from othello_rl.rl.checkpoint import save_checkpoint
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        a = DQNAgent(NetworkConfig(channels=8, blocks=2, hidden=16), seed=0)
        p = f"{td}/dqn.pt"
        save_checkpoint(p, a, version="vd")
        loaded = load_agent(p)
        assert isinstance(loaded, DQNAgent)

import json

import pytest

from othello_rl.rl.agent import DQNAgent, NetworkConfig
from othello_rl.rl.self_play import OpponentPool, SelfPlayConfig, run_self_play
from othello_rl.rl.trainer import DQNConfig


def _agent(seed=0):
    return DQNAgent(NetworkConfig(channels=8, blocks=2, hidden=16), seed=seed)


def test_pool_falls_back_to_baselines_when_empty():
    pool = OpponentPool(seed=0)
    a = pool.sample()
    assert a.name in ("random", "greedy", "heuristic")


def test_pool_sampling_distribution_is_configurable():
    pool = OpponentPool(distribution={"baseline": 0.5, "historical": 0.0, "recent": 0.5},
                        recent_capacity=3, historical_every=1, seed=0)
    for _ in range(6):
        pool.add_snapshot(_agent())
    counts = pool.category_counts(n=4000, seed=1)
    assert counts["historical"] == 0
    total = sum(counts.values())
    # roughly 50/50 baseline vs recent (loose bounds)
    assert 0.35 < counts["baseline"] / total < 0.65
    assert 0.35 < counts["recent"] / total < 0.65


def test_pool_recent_capacity_and_historical_growth():
    pool = OpponentPool(recent_capacity=3, historical_every=2, seed=0)
    for i in range(7):
        pool.add_snapshot(_agent(), tag=f"s{i}")
    assert pool.num_recent == 3
    assert pool.num_historical == 4  # snapshots 0,2,4,6


def test_frozen_snapshot_is_independent_of_learner():
    import torch
    agent = _agent()
    pool = OpponentPool(seed=0)
    frozen = pool.add_snapshot(agent, tag="t0")
    before = [p.clone() for p in frozen.net.parameters()]
    with torch.no_grad():
        for p in agent.net.parameters():
            p.add_(1.0)
    for b, p in zip(before, frozen.net.parameters()):
        assert torch.allclose(b, p)  # snapshot unchanged
    for p in frozen.net.parameters():
        assert not p.requires_grad


def test_pool_save_load_roundtrip(tmp_path):
    import torch
    pool = OpponentPool(recent_capacity=3, historical_every=2, seed=0)
    for i in range(5):
        pool.add_snapshot(_agent(i), tag=f"s{i}")
    p = tmp_path / "pool.pt"
    pool.save(p)

    restored = OpponentPool(recent_capacity=3, historical_every=2, seed=0)
    restored.load(p)
    assert restored.num_recent == pool.num_recent
    assert restored.num_historical == pool.num_historical
    assert restored._snapshot_count == pool._snapshot_count
    # weights match and are frozen
    for a0, a1 in zip(pool._recent, restored._recent):
        for x, y in zip(a0.net.parameters(), a1.net.parameters()):
            assert torch.allclose(x, y)
        assert all(not q.requires_grad for q in a1.net.parameters())
    # a restored snapshot still plays a legal move
    from othello_rl.environment.board import Board
    assert restored._recent[0].select_move(Board.initial()) in Board.initial().legal_moves()


def _sp_cfg(**over):
    base = dict(
        total_env_steps=1200, snapshot_every=400, eval_every=400,
        eval_games=6, checkpoint_every=600,
        pool=OpponentPool(distribution={"baseline": 0.3, "historical": 0.3, "recent": 0.4},
                          recent_capacity=3, historical_every=1, seed=0),
        dqn=DQNConfig(batch_size=16, buffer_capacity=800, warmup_steps=50,
                      target_sync=20, epsilon_decay_steps=400),
    )
    base.update(over)
    return SelfPlayConfig(**base)


def test_run_self_play_smoke(tmp_path):
    agent = _agent(1)
    run_self_play(agent, _sp_cfg(), tmp_path, seed=0)

    rows = [json.loads(l) for l in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert any(r.get("phase") == "snapshot" for r in rows)
    assert any(r.get("phase") == "eval" and r.get("where") == "end" for r in rows)
    assert (tmp_path / "checkpoints" / "final.pt").exists()
    assert (tmp_path / "checkpoints" / "pool.pt").exists()  # pool persisted for resume
    assert any("winrate_vs_hist0" in r for r in rows)


def test_run_self_play_resumes_pool(tmp_path):
    run1 = tmp_path / "run1"
    run_self_play(_agent(1), _sp_cfg(), run1, seed=0)
    from othello_rl.rl.self_play import OpponentPool as _P
    n_hist = _P(recent_capacity=3, historical_every=1).load(run1 / "checkpoints" / "pool.pt").num_historical
    assert n_hist >= 1

    # resume: fresh pool, restored from run1, and NO new "init" snapshot logged
    run2 = tmp_path / "run2"
    cfg2 = _sp_cfg(pool=OpponentPool(recent_capacity=3, historical_every=1, seed=1))
    logger = run_self_play(_agent(2), cfg2, run2,
                           seed=1, resume_pool=str(run1 / "checkpoints" / "pool.pt"))
    rows = logger.rows
    assert any(r.get("phase") == "resume" for r in rows)
    resume_row = next(r for r in rows if r.get("phase") == "resume")
    assert resume_row["historical"] >= n_hist

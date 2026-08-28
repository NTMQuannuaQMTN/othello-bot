import json

import numpy as np

from othello_rl.evaluation.harness import evaluate_agent, flatten_eval
from othello_rl.rl.agent import DQNAgent, NetworkConfig
from othello_rl.rl.curriculum import CurriculumConfig, Stage, run_curriculum
from othello_rl.rl.trainer import DQNConfig


def _agent():
    return DQNAgent(NetworkConfig(channels=16, blocks=2, hidden=32), seed=0)


def test_evaluate_agent_and_flatten():
    agent = _agent()
    res = evaluate_agent(agent, {"random": "random", "greedy": "greedy"},
                         num_games=8, seed=0)
    assert set(res) == {"random", "greedy"}
    for d in res.values():
        assert 0.0 <= d["win_rate"] <= 1.0
        assert d["ci_low"] <= d["win_rate"] <= d["ci_high"]
        assert d["wins"] + d["losses"] + d["draws"] == 8
    flat = flatten_eval(res)
    assert "winrate_vs_random" in flat and "disc_diff_vs_greedy" in flat


def test_run_curriculum_smoke(tmp_path):
    agent = _agent()
    cfg = CurriculumConfig(
        stages=[
            Stage("s1", "random", env_steps=400),
            Stage("s2", ["random", "greedy"], env_steps=400),
        ],
        eval_opponents={"random": "random"},
        eval_games=8,
        eval_every=200,
        checkpoint_every=300,
        dqn=DQNConfig(batch_size=16, buffer_capacity=1000, warmup_steps=50,
                      target_sync=25, epsilon_decay_steps=300),
    )
    logger = run_curriculum(agent, cfg, tmp_path, seed=0)

    metrics_file = tmp_path / "metrics.jsonl"
    assert metrics_file.exists()
    rows = [json.loads(l) for l in metrics_file.read_text().splitlines()]
    eval_rows = [r for r in rows if r.get("phase") == "eval"]
    assert any(r.get("stage") == "<untrained>" for r in eval_rows)
    assert any(r.get("stage") == "s1" for r in eval_rows)
    assert any(r.get("phase") == "done" for r in rows)

    # checkpoints written
    ckpts = list((tmp_path / "checkpoints").glob("*.pt"))
    names = {p.name for p in ckpts}
    assert "untrained.pt" in names
    assert "final.pt" in names
    assert "s1_final.pt" in names

    # final checkpoint reloads and plays legally
    reloaded = DQNAgent.from_checkpoint(tmp_path / "checkpoints" / "final.pt")
    assert reloaded.meta.env_steps >= 800
    res = evaluate_agent(reloaded, {"random": "random"}, num_games=4, seed=1)
    assert res["random"]["wins"] + res["random"]["losses"] + res["random"]["draws"] == 4

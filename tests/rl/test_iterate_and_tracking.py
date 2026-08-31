"""Phase 12.6 — experiment tracking, config-driven promotion, iterate skeleton."""
import hashlib
import json

import pytest
import torch

from othello_rl.rl.agent import DQNAgent, NetworkConfig
from othello_rl.rl.checkpoint import Registry, load_agent, save_checkpoint
from othello_rl.utils.experiment import log_experiment, read_experiments


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# experiment index
# --------------------------------------------------------------------------- #
def test_log_experiment_appends_and_persists(tmp_path):
    idx = tmp_path / "index.jsonl"
    log_experiment({"kind": "pretrain", "move_accuracy": 0.4}, path=idx)
    log_experiment({"kind": "eval", "win_rate": 0.9}, path=idx)
    rows = read_experiments(idx)
    assert [r["kind"] for r in rows] == ["pretrain", "eval"]
    assert all("timestamp" in r and "git_commit" in r for r in rows)
    # a fresh read (simulating a restart) still sees both
    assert len(read_experiments(idx)) == 2


# --------------------------------------------------------------------------- #
# Registry.promote is the only writer of production/ + registry.json
# --------------------------------------------------------------------------- #
def test_registry_promote_writes_production_and_registry(tmp_path):
    reg_path = tmp_path / "checkpoints" / "registry.json"
    reg_path.parent.mkdir(parents=True)
    (reg_path.parent / "production").mkdir()
    reg_path.write_text(json.dumps({"model_version": "v0",
                                    "checkpoint": "checkpoints/production/best.pt"}))
    cand = tmp_path / "cand.pt"
    save_checkpoint(cand, DQNAgent(NetworkConfig(channels=8, blocks=2, hidden=16), seed=1),
                    version="v1")

    reg = Registry.load(reg_path)
    out = reg.promote(cand, version="v1", parent="v0", method="test",
                      evaluation={"win_rate_vs_random": 0.9}, seed=1)
    assert out["model_version"] == "v1"
    best = reg_path.parent / "production" / "best.pt"
    latest = reg_path.parent / "production" / "latest.pt"
    assert _sha(best) == _sha(cand) == _sha(latest)
    assert json.loads(reg_path.read_text())["model_version"] == "v1"
    # a reload sees the promoted version
    assert Registry.load(reg_path).model_version == "v1"


# --------------------------------------------------------------------------- #
# promote_model.py — config-driven rule, rejection leaves production untouched
# --------------------------------------------------------------------------- #
def test_promote_model_config_rule_and_rejection(tmp_path, monkeypatch):
    import scripts.promote_model as pm
    from othello_rl.rl.checkpoint import REPO_ROOT

    best_src = REPO_ROOT / "checkpoints" / "production" / "best.pt"
    if not best_src.is_file():
        pytest.skip("no bundled production checkpoint")

    reg_path = tmp_path / "registry.json"
    reg_path.write_text(json.dumps({
        "model_version": "vbest", "checkpoint": str(best_src),
        "evaluation": {"win_rate_vs_random": 0.93, "win_rate_vs_greedy": 0.90}}))
    monkeypatch.setattr(Registry, "load", classmethod(
        lambda cls, p=None: Registry(json.loads(reg_path.read_text()), path=reg_path, exists=True)))
    idx = tmp_path / "index.jsonl"
    monkeypatch.setattr("othello_rl.utils.experiment.EXPERIMENT_INDEX", idx)
    monkeypatch.setattr(pm, "log_experiment",
                        lambda row, path=idx: log_experiment(row, path=idx))

    cfg = tmp_path / "promo.yaml"
    cfg.write_text("promotion:\n  min_vs_best_lb: 0.55\n  max_baseline_regression: 0.03\n"
                   "  min_games: 8\n")
    cand = tmp_path / "cand.pt"
    save_checkpoint(cand, DQNAgent(NetworkConfig(channels=8, blocks=2, hidden=16), seed=7),
                    version="vcand")
    sha_best = _sha(best_src)

    rc = pm.main([str(cand), "--config", str(cfg), "--games", "12", "--seed", "1"])
    assert rc == 1                                    # a fresh random net can't beat the bot
    assert _sha(best_src) == sha_best                 # production untouched
    assert json.loads(reg_path.read_text())["model_version"] == "vbest"
    rows = read_experiments(idx)
    assert rows and rows[-1]["kind"] == "promotion" and rows[-1]["decision"] == "rejected"


def test_promote_model_rejects_when_games_below_min(tmp_path, monkeypatch):
    import scripts.promote_model as pm
    monkeypatch.setattr(Registry, "load", classmethod(
        lambda cls, p=None: Registry({"model_version": "v0"}, path=tmp_path / "r.json", exists=True)))
    cfg = tmp_path / "p.yaml"
    cfg.write_text("promotion:\n  min_games: 500\n")
    cand = tmp_path / "c.pt"
    save_checkpoint(cand, DQNAgent(NetworkConfig(channels=8, blocks=2, hidden=16), seed=1))
    assert pm.main([str(cand), "--config", str(cfg), "--games", "50"]) == 2


# --------------------------------------------------------------------------- #
# iterate.py skeleton
# --------------------------------------------------------------------------- #
def test_iterate_dry_run_prints_the_plan(capsys):
    import scripts.iterate as it
    rc = it.main(["--dry-run", "--from", "analyze", "--to", "promote"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "plan: analyze -> dataset -> pretrain -> eval -> promote" in out
    assert "scripts/analyze_games.py" in out and "scripts/promote_model.py" in out


def test_eval_bot_load_agent_accepts_both_kinds(tmp_path):
    from othello_rl.rl.az_agent import PolicyValueAgent
    from othello_rl.rl.az_network import PolicyValueConfig
    from othello_rl.rl.checkpoint import save_policy_value_checkpoint

    dqn = tmp_path / "dqn.pt"
    save_checkpoint(dqn, DQNAgent(NetworkConfig(channels=8, blocks=2, hidden=16), seed=1))
    assert isinstance(load_agent(dqn), DQNAgent)

    pv_cfg = PolicyValueConfig(channels=8, blocks=2, hidden=16, norm=False)
    pv = tmp_path / "pv.pt"
    save_policy_value_checkpoint(pv, pv_cfg.build(), pv_cfg, version="vpv")
    assert isinstance(load_agent(pv), PolicyValueAgent)

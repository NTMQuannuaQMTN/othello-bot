"""Persistent checkpoint & production-model management
(``src/othello_rl/rl/checkpoint.py``)."""
import hashlib
import json

import numpy as np
import pytest
import torch

from othello_rl.environment.board import Board
from othello_rl.environment.environment import encode_observation, legal_action_mask
from othello_rl.rl.agent import DQNAgent, NetworkConfig
from othello_rl.rl.checkpoint import (
    Registry,
    load_checkpoint,
    resolve_checkpoint,
    restore_training,
    save_checkpoint,
)
from othello_rl.rl.curriculum import CurriculumConfig, Stage, run_curriculum
from othello_rl.rl.trainer import DQNConfig


def _agent(seed=0):
    return DQNAgent(NetworkConfig(channels=16, blocks=2, hidden=32), seed=seed)


def _boards(n=20):
    rng = np.random.RandomState(0)
    out, s = [], Board.initial()
    for _ in range(n):
        if s.is_terminal():
            s = Board.initial()
        moves = s.legal_moves()
        out.append(s)
        s = s.apply(moves[rng.randint(len(moves))] if moves else None)
    return out


# 1. save/load produces identical predictions --------------------------------
def test_save_load_identical_predictions(tmp_path):
    a = _agent(1)
    save_checkpoint(tmp_path / "c.pt", a, seed=1, version="vTEST")
    b = DQNAgent(device="cpu")
    rs = restore_training(tmp_path / "c.pt", b)
    assert rs.version == "vTEST"
    for board in _boards():
        obs, mask = encode_observation(board), legal_action_mask(board)
        q_a = a.q_values(obs, mask)
        q_b = b.q_values(obs, mask)
        np.testing.assert_array_equal(np.nan_to_num(q_a), np.nan_to_num(q_b))
        assert a.greedy_act(obs, mask) == b.greedy_act(obs, mask)


# 2. a "restarted backend" reloads the trained checkpoint (version + weights) -
def test_backend_restart_loads_trained_checkpoint(tmp_path):
    from othello_rl.webapp.bot_service import OthelloBot

    a = _agent(2)
    with torch.no_grad():                       # pretend this is a fine-tuned net
        for p in a.net.parameters():
            p.add_(0.01)
    a.meta.extra["version"] = 3
    a.meta.extra["games_finetuned"] = 3
    ckpt = tmp_path / "checkpoints" / "production" / "best.pt"
    save_checkpoint(ckpt, a, version="v003")
    (tmp_path / "checkpoints" / "registry.json").write_text(json.dumps({
        "model_version": "v003", "checkpoint": str(ckpt)}))

    reg = Registry.load(tmp_path / "checkpoints" / "registry.json")
    assert not reg.is_default()
    sd0 = None
    for _ in range(2):                           # two independent "boots"
        bot = OthelloBot.load(str(reg.active_checkpoint_path()))
        assert bot.version == 3
        sd = bot.agent.net.state_dict()
        if sd0 is None:
            sd0 = {k: v.clone() for k, v in sd.items()}
        for k, v in sd.items():
            assert torch.allclose(v, sd0[k])


# 3. constructing the web app / "rebuilding the frontend" never mutates a checkpoint
def test_appstate_does_not_touch_checkpoints(tmp_path):
    from othello_rl.webapp.server import AppState

    a = _agent(3)
    ckpt = tmp_path / "models" / "prod.pt"
    save_checkpoint(ckpt, a, version="v1")
    before = hashlib.sha256(ckpt.read_bytes()).hexdigest()

    bot = __import__("othello_rl.webapp.bot_service", fromlist=["OthelloBot"]).OthelloBot(
        a, source_path=str(ckpt), state_dir=str(tmp_path / "scratch"))
    app = AppState(bot, games_path=str(tmp_path / "data" / "games.jsonl"))
    app.session.new_game("black")
    app.session.human_move(app.session.state()["legal_actions"][0])
    (tmp_path / "web_dist").mkdir()
    (tmp_path / "web_dist" / "index.html").write_text("<!doctype html>")  # "rebuild"

    assert hashlib.sha256(ckpt.read_bytes()).hexdigest() == before


# 4. interrupted training resumes (counters + optimizer, not reset) ----------
def _tiny_cfg():
    return CurriculumConfig(
        stages=[Stage("s1", "random", env_steps=300)],
        eval_opponents={"random": "random"}, eval_games=4, eval_every=150,
        checkpoint_every=200,
        dqn=DQNConfig(batch_size=16, buffer_capacity=800, warmup_steps=40,
                      target_sync=25, epsilon_decay_steps=200))


def test_interrupted_training_resumes(tmp_path):
    a = _agent(4)
    run_curriculum(a, _tiny_cfg(), tmp_path / "r1", seed=0)
    ck1 = load_checkpoint(tmp_path / "r1" / "checkpoints" / "final.pt")
    assert ck1.meta.env_steps >= 300
    assert ck1.optimizer_state is not None and ck1.optimizer_state["state"]

    b = DQNAgent(device="cpu")
    rs = restore_training(tmp_path / "r1" / "checkpoints" / "final.pt", b)
    assert rs.env_steps == ck1.meta.env_steps and rs.train_steps == ck1.train_step
    run_curriculum(b, _tiny_cfg(), tmp_path / "r2", seed=0, resume_state=rs)
    ck2 = load_checkpoint(tmp_path / "r2" / "checkpoints" / "final.pt")
    assert ck2.meta.env_steps > ck1.meta.env_steps          # continued, not restarted
    assert not (tmp_path / "r2" / "checkpoints" / "untrained.pt").exists()


# 5 + 6. production model is protected from unevaluated / worse candidates ----
def test_registry_promote_is_the_only_writer(tmp_path):
    reg_path = tmp_path / "checkpoints" / "registry.json"
    a = _agent(5)
    best = tmp_path / "checkpoints" / "production" / "best.pt"
    save_checkpoint(best, a, version="v1")
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    reg_path.write_text(json.dumps({"model_version": "v1", "checkpoint": str(best),
                                    "evaluation": {"win_rate_vs_random": 0.9}}))
    sha_before = hashlib.sha256(best.read_bytes()).hexdigest()

    # merely loading / reading the registry changes nothing
    reg = Registry.load(reg_path)
    _ = reg.active_checkpoint_path()
    assert hashlib.sha256(best.read_bytes()).hexdigest() == sha_before
    assert json.loads(reg_path.read_text())["model_version"] == "v1"


def test_promote_model_rejects_a_worse_candidate(tmp_path, monkeypatch):
    import scripts.promote_model as pm

    # strong "best" = the bundled bot; weak candidate = a fresh random net
    from othello_rl.rl.checkpoint import REPO_ROOT
    best_src = REPO_ROOT / "checkpoints" / "production" / "best.pt"
    if not best_src.is_file():
        pytest.skip("no bundled production checkpoint")
    reg_path = tmp_path / "registry.json"
    reg_path.write_text(json.dumps({
        "model_version": "vbest", "checkpoint": str(best_src),
        "evaluation": {"win_rate_vs_random": 0.93, "win_rate_vs_greedy": 0.90}}))
    monkeypatch.setattr(Registry, "load", classmethod(lambda cls, p=None: Registry(
        json.loads(reg_path.read_text()), path=reg_path, exists=True)))

    cand = tmp_path / "cand.pt"
    save_checkpoint(cand, _agent(7), version="vcand")
    sha_best = hashlib.sha256(best_src.read_bytes()).hexdigest()

    rc = pm.main([str(cand), "--games", "16", "--seed", "1"])
    assert rc == 1                                          # criterion failed
    assert hashlib.sha256(best_src.read_bytes()).hexdigest() == sha_best
    assert json.loads(reg_path.read_text())["model_version"] == "vbest"


# 7. resolver + never-silent-V0 --------------------------------------------
def test_resolve_checkpoint_and_no_silent_v0(tmp_path):
    root = tmp_path / "checkpoints"
    (root / "production").mkdir(parents=True)
    (root / "production" / "best.pt").write_bytes(b"x")
    assert resolve_checkpoint("best", root=root) == root / "production" / "best.pt"
    assert resolve_checkpoint("latest", root=root) == root / "production" / "latest.pt"

    reg = Registry.load(tmp_path / "nope.json")
    assert reg.is_default()          # no registry file -> caller must warn (serve.py does)
    # the default still points at a concrete *initial* checkpoint, never a random net
    assert "initial" in str(reg.active_checkpoint_path())

    # a genuinely missing checkpoint is a hard failure, not a silent fresh model
    with pytest.raises(Exception):
        load_checkpoint(tmp_path / "does_not_exist.pt")

"""Smoke-test the CLI scripts end to end (tiny workloads)."""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _run(args, stdin=None, timeout=300):
    return subprocess.run([sys.executable, *args], cwd=ROOT, input=stdin,
                          capture_output=True, text=True, timeout=timeout)


def test_evaluate_script_runs(tmp_path):
    cfg = tmp_path / "eval.yaml"
    cfg.write_text(
        "seed: 1\nnum_games: 6\nopening_plies: 2\noutput_dir: %s\n"
        "matchups:\n  - [greedy, random]\n  - [heuristic, greedy]\n" % tmp_path
    )
    r = _run([str(SCRIPTS / "evaluate.py"), "--config", str(cfg)])
    assert r.returncode == 0, r.stderr
    runs = list(tmp_path.glob("*_eval"))
    assert runs and (runs[0] / "results.json").exists()
    assert (runs[0] / "report.md").exists()


def test_train_script_smoke(tmp_path):
    cfg = tmp_path / "train.yaml"
    cfg.write_text(
        "seed: 0\ndevice: cpu\ntag: smoke\n"
        "network: {channels: 16, blocks: 2, hidden: 32}\n"
        "dqn: {batch_size: 16, buffer_capacity: 800, warmup_steps: 40, "
        "target_sync: 25, epsilon_decay_steps: 200}\n"
        "eval: {opponents: {random: random}, games: 6, every: 150, seed: 5}\n"
        "checkpoint_every: 200\n"
        "stages:\n  - {name: s1, opponent: random, env_steps: 300, learner_color: random}\n"
    )
    r = _run([str(SCRIPTS / "train.py"), "--config", str(cfg), "--out", str(tmp_path)])
    assert r.returncode == 0, r.stderr
    runs = list(tmp_path.glob("*_smoke"))
    assert runs
    run = runs[0]
    assert (run / "metrics.jsonl").exists()
    assert (run / "checkpoints" / "final.pt").exists()
    assert (run / "summary.md").exists()


def test_track_script_smoke(tmp_path):
    from othello_rl.rl.agent import DQNAgent, NetworkConfig
    d = tmp_path / "run" / "checkpoints"
    d.mkdir(parents=True)
    for i in range(2):
        a = DQNAgent(NetworkConfig(channels=8, blocks=2, hidden=16), seed=i)
        a.meta.env_steps = i * 1000
        a.save(d / f"step{i * 1000}.pt")
    r = _run([str(SCRIPTS / "track.py"), "--run", str(tmp_path / "run"),
              "--games", "6", "--rr-games", "6", "--baselines", "random"])
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "run" / "tracking" / "tracking.md").exists()
    assert (tmp_path / "run" / "tracking" / "elo_vs_checkpoint.png").exists()


def test_selfplay_script_smoke(tmp_path):
    cfg = tmp_path / "sp.yaml"
    cfg.write_text(
        "seed: 0\ndevice: cpu\ntag: spsmoke\ninit_checkpoint: null\n"
        "network: {channels: 8, blocks: 2, hidden: 16}\n"
        "dqn: {batch_size: 16, buffer_capacity: 800, warmup_steps: 40, "
        "target_sync: 20, epsilon_decay_steps: 300}\n"
        "self_play: {total_env_steps: 900, snapshot_every: 300, eval_every: 300, "
        "eval_games: 6, checkpoint_every: 500, opening_plies: 2, "
        "pool: {distribution: {baseline: 0.4, historical: 0.2, recent: 0.4}, "
        "recent_capacity: 3, historical_every: 1}}\n"
    )
    r = _run([str(SCRIPTS / "selfplay.py"), "--config", str(cfg), "--out", str(tmp_path)])
    assert r.returncode == 0, r.stderr
    run = next(iter(tmp_path.glob("*_spsmoke")))
    assert (run / "checkpoints" / "final.pt").exists()
    assert (run / "metrics.jsonl").exists()


def test_serve_script_boots_and_serves(tmp_path):
    import json as _json
    import time as _time
    import urllib.request

    cfg = tmp_path / "web.yaml"
    cfg.write_text(
        f"host: 127.0.0.1\nport: 8771\ncheckpoint: models/othello_bot_v1.pt\n"
        f"state_dir: {tmp_path / 'state'}\nfinetune: {{grad_steps: 4, anchor_transitions: 60}}\n"
    )
    proc = subprocess.Popen([sys.executable, str(SCRIPTS / "serve.py"), "--config", str(cfg)],
                            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        ok = False
        for _ in range(50):
            try:
                with urllib.request.urlopen("http://127.0.0.1:8771/api/bot", timeout=1) as r:
                    info = _json.loads(r.read())
                    ok = info["params"] > 0
                    break
            except Exception:
                _time.sleep(0.3)
        assert ok, "server did not come up"
        with urllib.request.urlopen("http://127.0.0.1:8771/") as r:
            assert b"Othello" in r.read()
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_bot_cli_protocol():
    inp = "name\ngenmove\ngenmove f5d6c3\neval f5d6\nbogus\nquit\n"
    r = _run([str(SCRIPTS / "bot_cli.py"), "--checkpoint", "models/othello_bot_v1.pt"],
             stdin=inp)
    assert r.returncode == 0, r.stderr
    lines = r.stdout.strip().splitlines()
    assert lines[0].startswith("othello-bot")
    assert re.match(r"^[a-h][1-8]$", lines[1])          # opening move
    assert re.match(r"^[a-h][1-8]$", lines[2])
    assert re.match(r"^0\.\d+ [a-h][1-8]$", lines[3])   # eval line
    assert lines[4].startswith("? unknown")


def test_play_script_scripted_game():
    # Feed moves; game is short because we pass/へ play whatever is asked.
    # Use 'q' to quit after a couple of moves — exit code 0.
    moves = "\n".join(["d3", "q"]) + "\n"
    r = _run([str(SCRIPTS / "play.py"), "--opponent", "random", "--color", "black", "--seed", "1"],
             stdin=moves)
    assert r.returncode == 0, r.stderr
    assert "You are BLACK" in r.stdout

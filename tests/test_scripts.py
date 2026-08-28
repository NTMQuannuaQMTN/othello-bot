"""Smoke-test the CLI scripts end to end (tiny workloads)."""
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


def test_play_script_scripted_game():
    # Feed moves; game is short because we pass/へ play whatever is asked.
    # Use 'q' to quit after a couple of moves — exit code 0.
    moves = "\n".join(["d3", "q"]) + "\n"
    r = _run([str(SCRIPTS / "play.py"), "--opponent", "random", "--color", "black", "--seed", "1"],
             stdin=moves)
    assert r.returncode == 0, r.stderr
    assert "You are BLACK" in r.stdout

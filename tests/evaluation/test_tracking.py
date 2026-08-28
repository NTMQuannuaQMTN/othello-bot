import json

from othello_rl.evaluation.tracking import (
    discover_checkpoints,
    track_checkpoints,
    write_tracking_plots,
    write_tracking_report,
)
from othello_rl.rl.agent import DQNAgent, NetworkConfig


def _make_ckpts(tmp_path, n=3):
    d = tmp_path / "checkpoints"
    d.mkdir()
    for i in range(n):
        a = DQNAgent(NetworkConfig(channels=8, blocks=2, hidden=16), seed=i)
        a.meta.env_steps = i * 5000
        a.save(d / f"step{i * 5000}.pt")
    return tmp_path


def test_discover_and_track(tmp_path):
    run = _make_ckpts(tmp_path, n=3)
    paths = discover_checkpoints(run)
    assert len(paths) == 3

    result = track_checkpoints(paths, baselines={"random": "random"},
                               games=6, round_robin_games=6)
    assert result["elo_kind"] == "internal"
    assert len(result["checkpoints"]) == 3
    # sorted by env_steps
    steps = [e["env_steps"] for e in result["checkpoints"]]
    assert steps == sorted(steps)
    for e in result["checkpoints"]:
        assert "random" in e["vs"]
        assert 0.0 <= e["vs"]["random"]["win_rate"] <= 1.0
        assert "internal_elo" in e

    out = run / "tracking"
    write_tracking_report(result, out)
    write_tracking_plots(result, out)
    assert (out / "tracking.json").exists()
    assert (out / "tracking.md").exists()
    assert (out / "winrate_vs_checkpoint.png").exists()
    assert (out / "elo_vs_checkpoint.png").exists()
    loaded = json.loads((out / "tracking.json").read_text())
    assert loaded["checkpoints"][0]["name"].startswith("step")

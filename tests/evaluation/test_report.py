import json

from othello_rl.evaluation.report import generate_report
from othello_rl.evaluation.tournament import play_match


def test_generate_report_writes_files(tmp_path):
    matches = [
        play_match("greedy", "random", num_games=10, seed=1),
        play_match("heuristic", "random", num_games=10, seed=2),
    ]
    results = generate_report(matches, out_dir=tmp_path, seed=1)

    assert (tmp_path / "results.json").exists()
    assert (tmp_path / "report.md").exists()

    loaded = json.loads((tmp_path / "results.json").read_text())
    assert loaded["elo_kind"] == "internal"
    assert len(loaded["matches"]) == 2
    assert "greedy" in loaded["elo"] and "random" in loaded["elo"]

    md = (tmp_path / "report.md").read_text()
    assert "Internal" in md and "greedy" in md

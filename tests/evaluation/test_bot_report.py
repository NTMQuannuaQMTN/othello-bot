import json

from othello_rl.evaluation.bot_report import (
    STANDARD_PANEL,
    standard_panel_eval,
    write_bot_report,
)


def test_standard_panel_eval_structure_and_ordering(tmp_path):
    # small panel + few games to keep it fast; heuristic should beat random
    panel = {"random": "random", "greedy": "greedy", "minimax:1": "minimax:1"}
    res = standard_panel_eval("heuristic", agent_name="heuristic", panel=panel,
                              num_games=12, elo_extra_games=8, seed=1)
    assert res["agent"] == "heuristic"
    assert set(res["vs"]) == set(panel)
    for name, d in res["vs"].items():
        assert 0.0 <= d["win_rate"] <= 1.0
        assert d["ci_low"] <= d["win_rate"] <= d["ci_high"]
        assert d["wins"] + d["losses"] + d["draws"] == 12
    # heuristic clearly stronger than random
    assert res["vs"]["random"]["win_rate"] > 0.7
    assert res["elo_kind"] == "internal"
    assert "heuristic" in res["internal_elo"]
    # random is the anchor
    assert abs(res["internal_elo"]["random"] - 1500.0) < 1e-6

    md = write_bot_report(res, tmp_path)
    assert md.exists()
    assert (tmp_path / "bot_eval.json").exists()
    text = md.read_text()
    assert "Standard evaluation" in text and "Internal Elo" in text
    loaded = json.loads((tmp_path / "bot_eval.json").read_text())
    assert loaded["agent"] == "heuristic"


def test_standard_panel_default_has_minimax_depths():
    assert "minimax:1" in STANDARD_PANEL
    assert "minimax:2" in STANDARD_PANEL
    assert "minimax:3" in STANDARD_PANEL

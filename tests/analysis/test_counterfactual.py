import json
import random

from othello_rl.analysis import (analyze_file, analyze_game, benchmark, classify,
                                  judge_position, positions)
from othello_rl.analysis.counterfactual import _candidate_moves
from othello_rl.analysis.reconstruct import Position
from othello_rl.environment.board import BLACK, Board
from othello_rl.ingest.records import GameRecord
from tests.environment.conftest import make_board


def _game_record(seed, game_id="g"):
    rng = random.Random(seed)
    s = Board.initial()
    moves = []
    while not s.is_terminal():
        r, c = rng.choice(s.legal_moves())
        moves.append(r * 8 + c)
        s = s.apply((r, c))
    w = s.winner()
    return GameRecord(source="t", source_format="t", moves=moves, game_id=game_id,
                      result={"winner": "black" if w == 1 else "white" if w == -1 else "draw"})


# --------------------------------------------------------------------------- #
def test_classify_thresholds_and_config():
    assert classify(0.0) == "BEST"
    assert classify(0.04) == "GOOD"
    assert classify(0.10) == "ACCEPTABLE"
    assert classify(0.25) == "MISTAKE"
    assert classify(0.9) == "BLUNDER"
    tight = {"best": 0.0, "good": 0.0, "acceptable": 0.0, "mistake": 0.0}
    assert classify(0.001, tight) == "BLUNDER"     # every threshold configurable


def test_regret_is_best_minus_played_and_labels_the_played_move():
    b = make_board([".OX.....", ".O......", "..X.....", "...OX...",
                    "...XO...", "........", "........", "........"])
    st = Board(b, BLACK)
    legal = [r * 8 + c for r, c in st.legal_moves()]
    passive = next(m for m in legal if m != 0)

    def pos(played):
        return Position(st, BLACK, 4, played, legal, "g", "black", 30, 26)

    took_corner = judge_position(pos(0), 3)
    played_passive = judge_position(pos(passive), 3)

    assert took_corner.best_move == 0
    assert took_corner.regret >= -1e-9 and took_corner.label == "BEST"
    assert played_passive.regret > took_corner.regret
    order = ["BEST", "GOOD", "ACCEPTABLE", "MISTAKE", "BLUNDER"]
    assert order.index(played_passive.label) >= order.index(took_corner.label)
    assert played_passive.regret == round(
        played_passive.best_move_value - played_passive.played_move_value, 4)


def test_horizon_matters_somewhere():
    # at least one position must have a different best move at 1 vs 3 plies,
    # otherwise the horizon is meaningless.
    differ = 0
    for seed in range(3):
        for p in list(positions(_game_record(seed)))[:14]:
            if len(p.legal_moves) < 3:
                continue
            if judge_position(p, 1).best_move != judge_position(p, 3).best_move:
                differ += 1
    assert differ > 0


def test_max_alternatives_sampling_is_deterministic():
    rec = _game_record(2)
    p = next(pp for pp in positions(rec) if len(pp.legal_moves) >= 6)
    a, sampled_a = _candidate_moves(p, 4)
    b, sampled_b = _candidate_moves(p, 4)
    assert a == b and sampled_a and sampled_b and len(a) == 4
    assert p.played_move in a                       # played move always kept
    j1 = judge_position(p, 2, max_alternatives=4)
    j2 = judge_position(p, 2, max_alternatives=4)
    assert j1.as_dict() == j2.as_dict() and j1.sampled is True


def test_pipeline_perspective_and_shape():
    g = analyze_game(_game_record(3), {"lookahead_plies": [2, 3]})
    assert g["n_positions"] == len(g["moves"])
    for mv in g["moves"]:
        for h in ("2", "3"):
            j = mv["by_horizon"][h]
            assert j["label"] in ("BEST", "GOOD", "ACCEPTABLE", "MISTAKE", "BLUNDER")
            assert j["player"] == mv["player"]
            assert j["regret"] >= -1e-9 and j["lookahead_plies"] == int(h)


def test_analyze_file_and_stats(tmp_path):
    src = tmp_path / "valid.jsonl"
    src.write_text("\n".join(_game_record(s, f"g{s}").to_json() for s in range(3)))
    out = tmp_path / "analyzed.jsonl"
    stats = analyze_file(src, out, {"lookahead_plies": [2]}, report_dir=tmp_path / "rep")
    assert stats.games == 3 and stats.positions > 0
    games = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(games) == 3 and games[0]["game_id"] == "g0"
    assert sum(stats.label_counts["2"].values()) == stats.positions
    assert (tmp_path / "rep" / "analysis.stats.json").is_file()


def test_benchmark_reports_timing_per_horizon():
    b = benchmark([_game_record(s) for s in range(2)], {"lookahead_plies": [2, 3]})
    assert b["games"] == 2 and b["positions"] > 0 and b["mean_branching_factor"] > 1
    assert "horizon_2" in b and "horizon_3" in b
    assert b["horizon_2"]["positions_per_sec"] > 0

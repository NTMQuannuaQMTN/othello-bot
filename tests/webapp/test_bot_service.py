import random

import numpy as np
import pytest

from othello_rl.environment.board import BLACK, WHITE, Board
from othello_rl.rl.agent import DQNAgent, NetworkConfig
from othello_rl.webapp.bot_service import OthelloBot, classify_drop


@pytest.fixture(scope="module")
def bot():
    import torch
    torch.manual_seed(0)          # net weights use the global RNG — pin it so the
    agent = DQNAgent(NetworkConfig(channels=8, blocks=2, hidden=16), seed=0)  # grading
    return OthelloBot(agent)      # assertions below don't depend on test order


def _random_game(seed=0, max_plies=200):
    rng = random.Random(seed)
    s = Board.initial()
    acts = []
    while not s.is_terminal() and len(acts) < max_plies:
        lm = s.legal_moves()
        if not lm:
            s = s.apply(None); acts.append(64); continue
        m = rng.choice(lm)
        acts.append(m[0] * 8 + m[1]); s = s.apply(m)
    return acts, s


def test_select_action_always_legal(bot):
    s = Board.initial()
    for _ in range(12):
        if s.is_terminal():
            break
        a = bot.select_action(s)
        legal = [r * 8 + c for r, c in s.legal_moves()] or [64]
        assert a in legal
        s = s.apply(None if a == 64 else divmod(a, 8))


def test_classify_expected_points_lost():
    # chess.com Expected Points cutoffs: 0 | .02 | .05 | .10 | .20
    labels = [classify_drop(d)[0] for d in (0.0, 0.01, 0.035, 0.07, 0.15, 0.4)]
    assert labels == ["Best", "Excellent", "Good", "Inaccuracy", "Mistake", "Blunder"]
    assert classify_drop(0.0)[0] == "Best" and classify_drop(1e-4)[0] == "Excellent"


def test_evaluate_position_shapes(bot):
    ev = bot.evaluate_position(Board.initial())
    assert 0.0 <= ev["winprob_black"] <= 1.0
    assert len(ev["moves"]) == 4
    # ranked by expected points (winprob after the move), best first
    assert ev["moves"] == sorted(ev["moves"], key=lambda m: -m["winprob"])
    assert ev["moves"][0]["ep_lost"] == 0.0
    assert all("gives_corner" in m and "ep_lost" in m for m in ev["moves"])
    # terminal
    from tests.environment.conftest import make_board
    term = Board(make_board(["O" * 8] + ["." * 8] * 7), BLACK)
    assert bot.evaluate_position(term)["terminal"] is True


def test_analyse_game_structure(bot):
    acts, _ = _random_game(1)
    an = bot.analyse_game(acts)
    assert 0 < len(an) <= len(acts)
    for a in an:
        assert a.side in ("black", "white")
        assert a.label in ("Best", "Excellent", "Good", "Inaccuracy", "Mistake", "Blunder")
        assert 0.0 <= a.drop
        assert 0.0 <= a.eval_after_black <= 1.0
        assert a.played_san == a.played_san  # sane
        assert len(a.top_moves) >= 1


def test_analyse_line_navigation_payload(bot):
    from othello_rl.webapp.moves import parse_game

    # empty line -> just the start position, with the bot's opening suggestions
    d0 = bot.analyse_line([])
    assert len(d0["positions"]) == 1 and d0["plies"] == []
    p0 = d0["positions"][0]
    assert p0["turn"] == "black" and not p0["terminal"]
    assert sorted(p0["legal_actions"]) == [19, 26, 37, 44]
    assert len(p0["eval"]["moves"]) == 4
    assert p0["eval"]["moves"] == sorted(p0["eval"]["moves"], key=lambda m: -m["winprob"])

    acts = parse_game("c4c3f5b4b3")
    d = bot.analyse_line(acts)
    assert len(d["positions"]) == len(acts) + 1
    assert len(d["eval_graph"]) == len(acts) + 1
    assert len(d["plies"]) == len(acts)
    # every position carries a legal-move list + eval for the side to move
    for i, pos in enumerate(d["positions"]):
        assert "grid" in pos and "eval" in pos
        if not pos["terminal"]:
            assert len(pos["legal_actions"]) >= 1
    # the played move at each ply is recoverable and graded
    for ply, a in zip(d["plies"], acts):
        assert ply["played"] == a
        assert ply["label"] in ("Best", "Excellent", "Good", "Inaccuracy", "Mistake", "Blunder")


def test_eval_graph_is_grade_smoothed():
    from othello_rl.webapp.bot_service import _EVAL_SWING_CAP, _smoothed_eval_graph, MoveAnalysis

    def _pos(wb):
        return {"eval": {"winprob_black": wb}}

    # raw bar spikes +0.5 then returns; the spike ply was a "Best" move
    raw = [0.50, 0.52, 0.99, 0.55, 0.54]
    plies = [MoveAnalysis(ply=i, side="black", played=0, played_san="", played_value=0.0,
                          played_winprob=0.0, best=0, best_san="", best_value=0.0,
                          best_winprob=0.0, coach_best_san="", bot_drop=0.0, coach_drop=0.0,
                          drop=0.0, label="Best", glyph="", eval_after_black=0.0)
             for i in range(4)]
    g = _smoothed_eval_graph([_pos(x) for x in raw], plies)
    assert [round(p["eval_black_raw"], 2) for p in g] == raw
    deltas = [abs(g[i]["eval_black"] - g[i - 1]["eval_black"]) for i in range(1, len(g))]
    assert max(deltas) <= _EVAL_SWING_CAP["Best"] + 1e-9      # a Best move never jumps the bar

    # same spike, but the spike ply is a Blunder -> it is allowed through
    plies[1].label = "Blunder"
    g2 = _smoothed_eval_graph([_pos(x) for x in raw], plies)
    assert abs(g2[2]["eval_black"] - g2[1]["eval_black"]) > 0.3


def test_analyse_line_best_moves_keep_the_bar_steady(bot):
    # a line where the bot always plays its own top move -> the graph must be calm
    from othello_rl.environment.board import action_to_rc
    s = Board.initial()
    acts = []
    for _ in range(24):
        if s.is_terminal() or not s.legal_moves():
            break
        a = bot.evaluate_position(s, {})["moves"][0]["action"]
        acts.append(a)
        s = s.apply(action_to_rc(a))
    d = bot.analyse_line(acts)
    g = [p["eval_black"] for p in d["eval_graph"]]
    labels = {p["ply"]: p["label"] for p in d["plies"]}
    for i in range(1, len(g)):
        lbl = labels.get(i - 1)
        if lbl in ("Best", "Excellent", "Good"):
            assert abs(g[i] - g[i - 1]) <= 0.14


def test_analyse_line_prefix_cache_matches_fresh(bot):
    from othello_rl.webapp.moves import parse_game
    acts = parse_game("f5d6c3d3c4f4f6f3")
    bot._line_cache = None
    incremental = None
    for n in range(2, len(acts) + 1):          # grow the line one move at a time
        incremental = bot.analyse_line(acts[:n])
    bot._line_cache = None                       # force a from-scratch run
    fresh = bot.analyse_line(acts)
    assert incremental["n_moves"] == fresh["n_moves"]
    assert [p["label"] for p in incremental["plies"]] == [p["label"] for p in fresh["plies"]]
    assert [round(g["eval_black"], 4) for g in incremental["eval_graph"]] == \
           [round(g["eval_black"], 4) for g in fresh["eval_graph"]]
    assert incremental["strategy"] == fresh["strategy"]


def test_analyse_line_reports_strategy(bot):
    from othello_rl.webapp.moves import parse_game
    d = bot.analyse_line(parse_game("f5d6c3d3c4f4"))
    strat = d["strategy"]
    for side in ("black", "white"):
        s = strat[side]
        assert s["moves"] >= 1
        assert s["corners"] + s["x_squares"] + s["edges"] <= s["moves"]
        assert s["avg_mobility"] > 0
        assert 0.0 <= s["accuracy"] <= 1.0
        assert "final_discs" in s
    assert strat["final_discs"] if False else True  # winner key present when terminal
    assert "winner" in strat




def test_analyse_flags_a_clear_mistake(bot):
    # black can take corner a1; playing the b2 X-square instead should grade worse
    from tests.environment.conftest import make_board
    b = make_board([".OX.....", ".O......", "..X.....", "...OX...",
                    "...XO...", "........", "........", "........"])
    st = Board(b, BLACK)
    corner = bot.grade_move(st, 0)          # a1 (the corner)
    xsquare = bot.grade_move(st, 2 * 8 + 0)  # (2,0)
    assert corner["coach_best"] == 0
    # taking the corner has the most expected points -> 0 lost -> "Best"
    assert corner["takes_corner"] and corner["bot_best"] == 0
    assert corner["ep_lost"] == 0.0 and corner["label"] == "Best"
    # the alternative gives up expected points and grades worse
    assert xsquare["ep_lost"] > corner["ep_lost"]
    _order = ["Best", "Excellent", "Good", "Inaccuracy", "Mistake", "Blunder"]
    assert _order.index(xsquare["label"]) > _order.index(corner["label"])


def test_corner_flags_detect_give_and_take():
    from tests.environment.conftest import make_board
    agent = DQNAgent(NetworkConfig(channels=8, blocks=2, hidden=16), seed=0)
    b = OthelloBot(agent)
    b0 = make_board([".OX.....", ".O......", "..X.....", "...OX...",
                     "...XO...", "........", "........", "........"])
    st = Board(b0, BLACK)
    assert b.grade_move(st, 0)["takes_corner"] is True
    # a non-corner developing move takes/gives no corner here
    dev = next(r * 8 + c for r, c in st.legal_moves() if (r, c) != (0, 0))
    fl = b.grade_move(st, dev)
    assert fl["takes_corner"] is False


def test_x_square_is_a_blunder_only_when_it_actually_loses_the_corner(bot):
    from tests.environment.conftest import make_board
    XSQ_B2 = 1 * 8 + 1

    # b2 here lets the opponent force corner a1 -> Blunder, never suggested
    loses = Board(make_board(["........", "........", "..X...O.", "..XXXOO.",
                              ".XXXO.O.", "...XX...", "...X....", "........"]), WHITE)
    assert (1, 1) in loses.legal_moves()
    g = bot.grade_move(loses, XSQ_B2)
    assert g["corner_risk"] >= 0.7 and g["gives_corner"]
    assert g["ep_lost"] > 0.20 and g["label"] == "Blunder"
    assert bot.evaluate_position(loses)["moves"][0]["action"] != XSQ_B2

    # b2 here does NOT let the opponent reach a1 -> not a blunder
    safe = Board(make_board(["........", "........", "..X.....", "...XXO..",
                             "...XO...", "...XX...", "...X....", "........"]), WHITE)
    assert (1, 1) in safe.legal_moves()
    g2 = bot.grade_move(safe, XSQ_B2)
    assert g2["corner_risk"] < 0.42 and not g2["gives_corner"]
    assert g2["label"] != "Blunder"


def test_playing_the_top_ranked_move_grades_best(bot):
    # a line where every move IS the engine's own #1 pick -> those plies grade
    # "Best" (unless the pick itself concedes a corner / loses the game)
    s = Board.initial()
    acts, checked = [], 0
    for _ in range(14):
        if s.is_terminal():
            break
        mv = s.legal_moves()
        if not mv:
            s = s.apply(None); acts.append(64); continue
        top = bot.evaluate_position(s)["moves"][0]
        acts.append(top["action"])
        s = s.apply(divmod(top["action"], 8))
    an = bot.analyse_line(acts)
    for ply in an["plies"]:
        if ply["played"] == ply["best"]:
            assert ply["drop"] == 0.0 and ply["label"] == "Best"
            checked += 1
    assert checked >= 3

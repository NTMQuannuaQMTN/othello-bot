"""The bitboard Othello engine and the search behind ``OthelloBot.best_move``."""
from __future__ import annotations

import random
import time

import pytest

from othello_rl.engine import bitboard as bb
from othello_rl.engine.solver import best_move, evaluate
from othello_rl.environment.board import Board, action_to_rc, rc_to_action


# --------------------------------------------------------------------------- #
# bitboard == the validated numpy engine
# --------------------------------------------------------------------------- #
def test_bitboard_matches_numpy_engine():
    rng = random.Random(0)
    for game in range(120):
        b = Board.initial()
        for _ in range(64):
            if b.is_terminal():
                break
            P, O = bb.from_grid(b.array, b.player)
            np_moves = {rc_to_action(r, c) for r, c in b.legal_moves()}
            bb_moves = set(bb.move_list(bb.legal_moves(P, O)))
            assert np_moves == bb_moves, (game, sorted(np_moves), sorted(bb_moves))
            if not np_moves:
                b = b.apply(None)
                continue
            mv = rng.choice(sorted(np_moves))
            nb = b.apply(action_to_rc(mv))
            nP, nO = bb.play(P, O, mv)
            got = (nO, nP) if nb.player == b.player else (nP, nO)   # nb may auto-pass
            assert got == bb.from_grid(nb.array, nb.player)
            b = nb
        if b.is_terminal():
            P, O = bb.from_grid(b.array, b.player)
            fs, wn = bb.final_score(P, O), b.winner()
            assert (fs > 0) == (wn == b.player) and (fs < 0) == (wn == -b.player)


# --------------------------------------------------------------------------- #
# exact endgame solve == brute-force negamax
# --------------------------------------------------------------------------- #
def _brute(P, O):
    my = bb.legal_moves(P, O)
    if my == 0:
        return bb.final_score(P, O) if bb.legal_moves(O, P) == 0 else -_brute(O, P)
    return max(-_brute(*bb.play(P, O, s)) for s in bb.move_list(my))


def test_exact_endgame_is_optimal():
    rng = random.Random(2)
    checked = 0
    for _ in range(120):
        b = Board.initial()
        while not b.is_terminal() and int((b.array != 0).sum()) < 57:  # -> ~7 empties
            lm = b.legal_moves()
            b = b.apply(rng.choice(lm) if lm else None)
        if b.is_terminal():
            continue
        P, O = bb.from_grid(b.array, b.player)
        e = 64 - bb.popcount(P) - bb.popcount(O)
        sq, val, meta = best_move(P, O, endgame_empties=e, time_budget=10.0)
        assert meta["exact"]
        margin = (val - (2 ** 20 if val > 0 else -2 ** 20 if val < 0 else 0)) / 1000
        assert round(margin) == _brute(P, O)
        # the move it recommends really achieves that value
        assert -_brute(*bb.play(P, O, sq)) == _brute(P, O)
        checked += 1
    assert checked >= 20


# --------------------------------------------------------------------------- #
# midgame search
# --------------------------------------------------------------------------- #
def test_midgame_search_deepens_and_is_legal():
    rng = random.Random(3)
    b = Board.initial()
    for _ in range(14):
        lm = b.legal_moves()
        b = b.apply(rng.choice(lm) if lm else None)
    P, O = bb.from_grid(b.array, b.player)
    t0 = time.monotonic()
    sq, val, meta = best_move(P, O, time_budget=1.5)
    assert time.monotonic() - t0 < 3.0
    assert meta["depth"] >= 3
    assert (bb.legal_moves(P, O) >> sq) & 1
    assert meta["pv"] and meta["pv"][0] == sq


# --------------------------------------------------------------------------- #
# OthelloBot integration
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def bot():
    from othello_rl.rl.checkpoint import Registry
    from othello_rl.webapp.bot_service import OthelloBot
    b = OthelloBot.load(str(Registry.load().active_checkpoint_path()))
    b.engine_budget = 0.4
    b.engine_endgame = 10
    return b


def test_best_move_payload(bot):
    from othello_rl.environment.board import PASS_ACTION
    d = bot.best_move(Board.initial())
    assert 0 <= d["action"] < 64
    assert action_to_rc(d["action"]) in Board.initial().legal_moves()
    assert 0.0 <= d["winprob"] <= 1.0 and d["depth"] >= 1
    # a forced-pass position
    forced = Board(_forced_pass_grid(), 1)
    assert bot.best_move(forced)["action"] == PASS_ACTION


def test_evaluate_position_top_move_is_the_engine_move(bot):
    from othello_rl.webapp.moves import parse_game
    b = Board.initial()
    for mv in parse_game("c4c3f5b4b3"):
        b = b.apply(mv)
    ev = bot.evaluate_position(b)
    eng = bot.best_move(b)
    assert ev["moves"][0]["action"] == eng["action"]
    assert "engine" in ev


def test_engine_beats_a_depth2_minimax(bot):
    from othello_rl.agents.minimax_agent import MinimaxAgent
    opp = MinimaxAgent(depth=2)
    wins = 0
    for g in range(6):
        b = Board.initial()
        eng_black = g % 2 == 0
        while not b.is_terminal():
            if not b.legal_moves():
                b = b.apply(None)
                continue
            if (b.player == 1) == eng_black:
                b = b.apply(action_to_rc(bot.best_move(b)["action"]))
            else:
                b = b.apply(opp.select_move(b))
        sb, sw = b.scores()
        mine = sb if eng_black else sw
        theirs = sw if eng_black else sb
        wins += mine > theirs
    assert wins >= 5           # the search engine should crush a 2-ply minimax


def _forced_pass_grid():
    import numpy as np
    a = np.full((8, 8), -1, dtype=np.int8)   # all white ...
    a[0, 0] = 0                              # ... one empty corner, no black disc
    return a

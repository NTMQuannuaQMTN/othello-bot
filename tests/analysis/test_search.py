import random

import pytest

from othello_rl.agents.minimax_agent import MinimaxAgent
from othello_rl.analysis.search import move_value, shallow_value
from othello_rl.environment.board import BLACK, Board
from tests.environment.conftest import make_board


def _positions(seed, k):
    rng = random.Random(seed)
    s = Board.initial()
    out = []
    while not s.is_terminal() and len(out) < k:
        out.append(s)
        s = s.apply(rng.choice(s.legal_moves()))
    return out


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_shallow_value_equals_minimax_agent(depth):
    mm = MinimaxAgent(depth=depth)
    for s in _positions(depth, 6):
        ref = mm._value(s, depth, -float("inf"), float("inf"), prune=False)
        assert shallow_value(s, depth, alpha_beta=False) == pytest.approx(ref)
        # alpha-beta returns the same exact value at a full window
        assert shallow_value(s, depth, alpha_beta=True) == pytest.approx(ref)


@pytest.mark.parametrize("depth", [2, 3])
def test_alpha_beta_picks_the_same_move_as_plain_minimax(depth):
    for s in _positions(depth * 3, 6):
        legal = [r * 8 + c for r, c in s.legal_moves()]
        ab = {m: move_value(s, divmod(m, 8), depth, alpha_beta=True) for m in legal}
        plain = {m: move_value(s, divmod(m, 8), depth, alpha_beta=False) for m in legal}
        assert max(ab, key=ab.get) == max(plain, key=plain.get)
        for m in legal:
            assert ab[m] == pytest.approx(plain[m])


def test_transposition_table_does_not_change_values():
    for s in _positions(1, 6):
        a = shallow_value(s, 4, tt=None)
        b = shallow_value(s, 4, tt={})
        assert a == pytest.approx(b)


def test_move_value_is_from_the_movers_perspective():
    # black can take corner a1 here (row0: . O X ...)
    b = make_board([".OX.....", ".O......", "..X.....", "...OX...",
                    "...XO...", "........", "........", "........"])
    st = Board(b, BLACK)
    corner = move_value(st, (0, 0), 3)          # take a1
    xsq = move_value(st, (2, 0), 3)             # a C/X-ish square instead
    assert corner > xsq                          # corner is better *for black*


def test_search_handles_a_forced_pass_position():
    # find a real position where a player is skipped (Board.apply auto-passes),
    # rebuild that side as the side-to-move, and check shallow_value doesn't
    # crash and alternates correctly through the pass.
    from othello_rl.environment.board import opponent
    for seed in range(200):
        rng = random.Random(seed)
        s = Board.initial()
        while not s.is_terminal():
            nxt = s.apply(rng.choice(s.legal_moves()))
            if nxt.player == s.player and not nxt.is_terminal():
                stuck = Board(nxt.array, opponent(nxt.player))
                assert stuck.must_pass()
                v_ab = shallow_value(stuck, 3, alpha_beta=True)
                v_plain = shallow_value(stuck, 3, alpha_beta=False)
                assert v_ab == pytest.approx(v_plain)
                return
            s = nxt
    pytest.skip("no auto-pass position found in 200 seeds")

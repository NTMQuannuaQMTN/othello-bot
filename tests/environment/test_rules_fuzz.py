"""Randomised playouts: the engine must never crash and invariants must hold."""
import random

import numpy as np

from othello_rl.environment.board import BLACK, WHITE, Board
from othello_rl.environment import rules


def _check_apply_delta(before: Board, move, after: Board):
    """After a placing move, the only changed cells are {move} ∪ flipped, all set
    to the mover; disc counts move by exactly (+flips+1 / -flips)."""
    if move is None:
        assert np.array_equal(before.array, after.array), "pass changed the board"
        return
    r, c = move
    flips = rules.flips_for_move(before.array, before.player, move)
    changed = set(zip(*np.nonzero(before.array != after.array)))
    assert changed == {(r, c)} | set(flips), f"unexpected changed cells {changed}"
    for rr, cc in changed:
        assert after.array[rr, cc] == before.player
    mover = before.player
    m0 = int(np.sum(before.array == mover)); o0 = int(np.sum(before.array == -mover))
    m1 = int(np.sum(after.array == mover));  o1 = int(np.sum(after.array == -mover))
    assert m1 == m0 + len(flips) + 1, "mover disc count wrong"
    assert o1 == o0 - len(flips), "opponent disc count wrong"


def play_random_game(seed):
    rng = random.Random(seed)
    st = Board.initial()
    plies = 0
    passes_in_a_row = 0
    while not st.is_terminal():
        actions = st.legal_actions()
        assert actions, "non-terminal state must offer at least one action"
        moves = st.legal_moves()
        if moves:
            passes_in_a_row = 0
            # every legal move must actually be legal and flip >=1 disc
            for m in moves:
                assert rules.flips_for_move(st.array, st.player, m)
            choice = rng.choice(moves)
        else:
            passes_in_a_row += 1
            assert passes_in_a_row <= 1, "two passes in a row => should be terminal"
            choice = None
        prev_discs = int(np.sum(st.array != 0))
        before = st
        st = st.apply(choice)
        _check_apply_delta(before, choice, st)
        new_discs = int(np.sum(st.array != 0))
        if choice is None:
            assert new_discs == prev_discs
        else:
            assert new_discs == prev_discs + 1, "exactly one new disc per placing move"
        # the side to move always either has a move or the game is over
        assert st.is_terminal() or rules.has_any_move(st.array, st.player)
        plies += 1
        assert plies <= 120, "game exceeded the 120-ply Othello maximum"
    return st


def test_many_random_games():
    for seed in range(200):
        st = play_random_game(seed)
        b, w = st.scores()
        assert 0 <= b + w <= 64
        assert st.winner() in (BLACK, WHITE, 0)


def test_full_game_disc_count_conserved_or_grows():
    # In Othello total discs is non-decreasing and equals 4 + number of placing moves.
    rng = random.Random(42)
    st = Board.initial()
    placing_moves = 0
    while not st.is_terminal():
        moves = st.legal_moves()
        if moves:
            st = st.apply(rng.choice(moves))
            placing_moves += 1
        else:
            st = st.apply(None)
    b, w = st.scores()
    assert b + w == 4 + placing_moves


def test_apply_does_not_mutate_source():
    st = Board.initial()
    snapshot = st.array.copy()
    _ = st.apply((2, 3))
    assert np.array_equal(st.array, snapshot)

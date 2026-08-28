"""Randomised playouts: the engine must never crash and invariants must hold."""
import random

import numpy as np

from othello_rl.environment.board import BLACK, WHITE, Board
from othello_rl.environment import rules


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
        st = st.apply(choice)
        new_discs = int(np.sum(st.array != 0))
        if choice is None:
            assert new_discs == prev_discs
        else:
            assert new_discs == prev_discs + 1, "exactly one new disc per placing move"
        # the side to move always either has a move or the game is over
        assert st.is_terminal() or rules.has_any_move(st.array, st.player)
        plies += 1
        assert plies < 200, "game ran impossibly long"
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

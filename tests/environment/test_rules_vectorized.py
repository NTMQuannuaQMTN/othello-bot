"""The vectorised move generator must agree with a brute-force scalar reference."""
import random

import numpy as np

from othello_rl.environment.board import BLACK, WHITE, Board
from othello_rl.environment import rules


def _scalar_legal_moves(board, player):
    out = []
    for r in range(8):
        for c in range(8):
            if board[r, c] == 0 and rules.flips_for_move(board, player, (r, c)):
                out.append((r, c))
    return out


def test_vectorized_matches_scalar_on_random_positions():
    rng = random.Random(0)
    checked = 0
    for seed in range(60):
        st = Board.initial()
        r = random.Random(seed)
        while not st.is_terminal():
            for player in (BLACK, WHITE):
                assert rules.legal_moves(st.array, player) == _scalar_legal_moves(st.array, player)
                assert rules.has_any_move(st.array, player) == bool(_scalar_legal_moves(st.array, player))
                checked += 1
            moves = st.legal_moves()
            st = st.apply(r.choice(moves) if moves else None)
    assert checked > 500


def test_vectorized_matches_scalar_on_dense_random_boards():
    rng = np.random.default_rng(42)
    for _ in range(200):
        arr = rng.integers(-1, 2, size=(8, 8)).astype(np.int8)
        for player in (BLACK, WHITE):
            assert rules.legal_moves(arr, player) == _scalar_legal_moves(arr, player)

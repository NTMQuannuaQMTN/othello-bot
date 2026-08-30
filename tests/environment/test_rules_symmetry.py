"""Othello is invariant under the dihedral group of the square. If legal-move
generation had a wrong direction/offset, rotating/reflecting the board would
change the (transformed) legal-move set — this test would catch it."""
import random

import numpy as np

from othello_rl.environment.board import BLACK, WHITE, Board
from othello_rl.environment import rules


def _rot_point(r, c, k):
    for _ in range(k % 4):
        r, c = c, 7 - r
    return r, c


def _random_position(seed):
    rng = random.Random(seed)
    st = Board.initial()
    for _ in range(rng.randint(0, 45)):
        moves = st.legal_moves()
        if not moves:
            break
        st = st.apply(rng.choice(moves))
    return st


def test_legal_moves_are_dihedral_equivariant():
    for seed in range(80):
        st = _random_position(seed)
        for player in (BLACK, WHITE):
            base = set(rules.legal_moves(st.array, player))
            for k in range(4):
                rotated = np.ascontiguousarray(np.rot90(st.array, -k))
                got = set(rules.legal_moves(rotated, player))
                want = {_rot_point(r, c, k) for (r, c) in base}
                assert got == want, f"rot {k}: {got ^ want}"
            # horizontal reflection
            reflected = np.ascontiguousarray(st.array[:, ::-1])
            got = set(rules.legal_moves(reflected, player))
            want = {(r, 7 - c) for (r, c) in base}
            assert got == want, f"reflect: {got ^ want}"


def test_flip_sets_are_dihedral_equivariant():
    for seed in range(40):
        st = _random_position(seed + 100)
        player = st.player
        for m in st.legal_moves():
            base_flips = set(rules.flips_for_move(st.array, player, m))
            rotated = np.ascontiguousarray(np.rot90(st.array, -1))
            rm = _rot_point(*m, 1)
            got = set(rules.flips_for_move(rotated, player, rm))
            want = {_rot_point(r, c, 1) for (r, c) in base_flips}
            assert got == want

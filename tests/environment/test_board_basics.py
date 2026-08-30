import numpy as np
import pytest

from othello_rl.environment import board as B
from othello_rl.environment.board import Board


def test_initial_board_shape_and_discs():
    arr = B.initial_board()
    assert arr.shape == (8, 8)
    assert arr.dtype == np.int8
    assert arr[3, 3] == B.WHITE
    assert arr[4, 4] == B.WHITE
    assert arr[3, 4] == B.BLACK
    assert arr[4, 3] == B.BLACK
    assert np.sum(arr != 0) == 4


def test_black_moves_first():
    assert Board.initial().player == B.BLACK


def test_initial_legal_moves():
    st = Board.initial()
    assert sorted(st.legal_moves()) == [(2, 3), (3, 2), (4, 5), (5, 4)]


def test_coordinate_helpers_roundtrip():
    for a in range(64):
        r, c = B.action_to_rc(a)
        assert B.rc_to_action(r, c) == a
    assert B.action_to_rc(B.PASS_ACTION) is None


def test_square_name_parsing():
    assert B.parse_square("d3") == (2, 3)
    assert B.parse_square("a1") == (0, 0)
    assert B.parse_square("h8") == (7, 7)
    assert B.square_name((2, 3)) == "d3"
    assert B.square_name((7, 7)) == "h8"
    with pytest.raises(ValueError):
        B.parse_square("z9")


def test_opponent():
    assert B.opponent(B.BLACK) == B.WHITE
    assert B.opponent(B.WHITE) == B.BLACK
    with pytest.raises(ValueError):
        B.opponent(0)


def test_board_array_is_readonly():
    st = Board.initial()
    with pytest.raises(ValueError):
        st.array[0, 0] = 1


def test_board_does_not_alias_or_freeze_callers_array():
    # regression: Board must take an owned copy, not wrap/freeze the input
    arr = B.initial_board()
    st = Board(arr, B.BLACK)
    assert arr.flags.writeable, "constructing a Board froze the caller's array"
    arr[0, 0] = B.BLACK  # mutating the source must not affect the Board
    assert st.array[0, 0] == B.EMPTY
    # a second Board from a shared source array is independent
    src = B.initial_board()
    b1, b2 = Board(src, B.BLACK), Board(src, B.WHITE)
    src[7, 7] = B.WHITE
    assert b1.array[7, 7] == B.EMPTY and b2.array[7, 7] == B.EMPTY
    assert b1.array is not b2.array


def test_board_accepts_non_int8_input():
    arr = np.zeros((8, 8), dtype=np.int64)
    arr[3, 4] = 1
    st = Board(arr, B.BLACK)
    assert st.array.dtype == np.int8 and st.array[3, 4] == B.BLACK


def test_board_equality_and_hash():
    a = Board.initial()
    b = Board.initial()
    assert a == b
    assert hash(a) == hash(b)
    c = a.apply((2, 3))
    assert c != a

import numpy as np

from othello_rl.environment.board import BLACK, WHITE
from othello_rl.environment import rules
from .conftest import make_board

EMPTY_ROW = "........"


def test_horizontal_capture():
    board = make_board([EMPTY_ROW, EMPTY_ROW, EMPTY_ROW,
                        ".XOOO...",
                        EMPTY_ROW, EMPTY_ROW, EMPTY_ROW, EMPTY_ROW])
    flips = rules.flips_for_move(board, BLACK, (3, 5))
    assert sorted(flips) == [(3, 2), (3, 3), (3, 4)]
    new = rules.apply_move(board, BLACK, (3, 5))
    assert list(new[3, 1:6]) == [BLACK] * 5


def test_vertical_capture():
    board = make_board(["...X....", "...O....", "...O....",
                        EMPTY_ROW, EMPTY_ROW, EMPTY_ROW, EMPTY_ROW, EMPTY_ROW])
    flips = rules.flips_for_move(board, BLACK, (3, 3))
    assert sorted(flips) == [(1, 3), (2, 3)]


def test_diagonal_capture():
    board = make_board([".X......", "..O.....", "...O....",
                        EMPTY_ROW, EMPTY_ROW, EMPTY_ROW, EMPTY_ROW, EMPTY_ROW])
    flips = rules.flips_for_move(board, BLACK, (3, 4))
    assert sorted(flips) == [(1, 2), (2, 3)]


def test_multi_direction_single_move():
    # Black plays (4,4); flips run left, up, and up-left simultaneously.
    board = make_board([
        "........",
        ".X..X...",
        "..O.O...",
        "...OO...",
        ".XOO....",
        "........",
        "........",
        "........",
    ])
    flips = rules.flips_for_move(board, BLACK, (4, 4))
    assert sorted(flips) == sorted([(4, 2), (4, 3), (3, 4), (2, 4), (2, 2), (3, 3)])
    assert len(flips) == 6


def test_corner_move():
    board = make_board([".OX.....", EMPTY_ROW, EMPTY_ROW, EMPTY_ROW,
                        EMPTY_ROW, EMPTY_ROW, EMPTY_ROW, EMPTY_ROW])
    flips = rules.flips_for_move(board, BLACK, (0, 0))
    assert flips == [(0, 1)]


def test_edge_move():
    board = make_board(["...XO...", EMPTY_ROW, EMPTY_ROW, EMPTY_ROW,
                        EMPTY_ROW, EMPTY_ROW, EMPTY_ROW, EMPTY_ROW])
    # Black plays (0,5): bracket (0,4)=O between (0,5) and (0,3)=X.
    flips = rules.flips_for_move(board, BLACK, (0, 5))
    assert flips == [(0, 4)]


def test_capture_many_pieces_one_line():
    board = make_board([EMPTY_ROW, EMPTY_ROW, EMPTY_ROW,
                        "XOOOOOO.",
                        EMPTY_ROW, EMPTY_ROW, EMPTY_ROW, EMPTY_ROW])
    flips = rules.flips_for_move(board, BLACK, (3, 7))
    assert sorted(flips) == [(3, c) for c in range(1, 7)]


def test_white_to_move_capture():
    board = make_board([EMPTY_ROW, EMPTY_ROW, EMPTY_ROW,
                        ".OXXX...",
                        EMPTY_ROW, EMPTY_ROW, EMPTY_ROW, EMPTY_ROW])
    flips = rules.flips_for_move(board, WHITE, (3, 5))
    assert sorted(flips) == [(3, 2), (3, 3), (3, 4)]


def test_legal_moves_sorted_and_complete():
    board = make_board([EMPTY_ROW, EMPTY_ROW, "...OX...", "...XO...",
                        EMPTY_ROW, EMPTY_ROW, EMPTY_ROW, EMPTY_ROW])
    moves = rules.legal_moves(board, BLACK)
    assert moves == sorted(moves)
    # each reported move must actually flip something
    for m in moves:
        assert rules.flips_for_move(board, BLACK, m)

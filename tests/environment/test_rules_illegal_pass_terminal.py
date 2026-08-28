import numpy as np
import pytest

from othello_rl.environment.board import BLACK, WHITE, Board
from othello_rl.environment import board as B
from othello_rl.environment import rules
from .conftest import make_board

EMPTY_ROW = "........"


# --------------------------------------------------------------------------- #
# Illegal moves
# --------------------------------------------------------------------------- #
def test_illegal_occupied_square():
    st = Board.initial()
    assert not st.is_legal((3, 3))
    with pytest.raises(ValueError):
        rules.apply_move(st.array, BLACK, (3, 3))


def test_illegal_no_capture():
    st = Board.initial()
    assert not st.is_legal((0, 0))
    with pytest.raises(ValueError):
        rules.apply_move(st.array, BLACK, (0, 0))


def test_illegal_out_of_range():
    board = B.initial_board()
    assert rules.flips_for_move(board, BLACK, (8, 8)) == []
    assert rules.flips_for_move(board, BLACK, (-1, 0)) == []
    assert not rules.is_legal_move(board, BLACK, (8, 0))
    with pytest.raises(ValueError):
        B.rc_to_action(8, 0)
    with pytest.raises(ValueError):
        B.action_to_rc(100)


def test_illegal_pass_when_move_available():
    st = Board.initial()
    assert not st.is_legal(None)
    with pytest.raises(ValueError):
        rules.apply_move(st.array, BLACK, None)


# --------------------------------------------------------------------------- #
# Pass behaviour
# --------------------------------------------------------------------------- #
PASS_BOARD = make_board(["OX......", EMPTY_ROW, EMPTY_ROW, EMPTY_ROW,
                         EMPTY_ROW, EMPTY_ROW, EMPTY_ROW, EMPTY_ROW])


def test_pass_scenario_black_stuck_white_can_move():
    # (0,0)=O, (0,1)=X, rest empty. Black to move has no legal move; White does.
    assert rules.legal_moves(PASS_BOARD, BLACK) == []
    assert rules.legal_moves(PASS_BOARD, WHITE) == [(0, 2)]
    assert not rules.is_terminal(PASS_BOARD)

    st = Board(PASS_BOARD, BLACK)
    assert st.must_pass()
    assert st.legal_actions() == [B.PASS_ACTION]

    after_pass = st.apply(None)
    assert after_pass.player == WHITE
    assert np.array_equal(after_pass.array, PASS_BOARD)


def test_white_cannot_pass_here():
    st = Board(PASS_BOARD, WHITE)
    assert not st.must_pass()
    with pytest.raises(ValueError):
        st.apply(None)


def test_next_player_consecutive_move_and_normal():
    # Board full except (0,0); only Black can play there.
    arr = np.full((8, 8), BLACK, dtype=np.int8)
    arr[0, 0] = 0
    arr[0, 1] = WHITE
    arr[1, 0] = WHITE
    arr[1, 1] = WHITE
    arr[0, 2] = BLACK  # bracket for Black playing (0,0) going east
    assert rules.has_any_move(arr, BLACK) is True
    assert rules.has_any_move(arr, WHITE) is False
    assert rules.legal_moves(arr, BLACK) == [(0, 0)]
    # White just moved -> Black plays next (normal alternation)
    assert rules.next_player(arr, WHITE) == BLACK
    # Black just moved -> White must pass, Black continues
    assert rules.next_player(arr, BLACK) == BLACK
    assert not rules.is_terminal(arr)


# --------------------------------------------------------------------------- #
# Termination / scoring / winner
# --------------------------------------------------------------------------- #
def test_terminal_full_board():
    arr = np.full((8, 8), BLACK, dtype=np.int8)
    arr[:3, :] = WHITE  # 24 white, 40 black
    assert rules.is_terminal(arr)
    assert rules.score(arr) == (40, 24)
    assert rules.winner(arr) == BLACK


def test_terminal_no_moves_not_full():
    arr = make_board(["OOOOOOOO", EMPTY_ROW, EMPTY_ROW, EMPTY_ROW,
                      EMPTY_ROW, EMPTY_ROW, EMPTY_ROW, EMPTY_ROW])
    assert rules.is_terminal(arr)
    assert rules.score(arr) == (0, 8)
    assert rules.winner(arr) == WHITE


def test_draw_detection():
    arr = np.full((8, 8), BLACK, dtype=np.int8)
    arr[:4, :] = WHITE  # 32-32
    assert rules.is_terminal(arr)
    assert rules.winner(arr) == 0


def test_board_apply_updates_scores():
    st = Board.initial()
    assert st.scores() == (2, 2)
    st2 = st.apply((2, 3))  # black plays d3, flips one
    assert st2.scores() == (4, 1)
    assert st2.player == WHITE

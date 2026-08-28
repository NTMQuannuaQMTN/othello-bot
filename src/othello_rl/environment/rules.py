"""Pure Othello rules operating on raw ``numpy`` boards.

All functions treat the board as immutable: :func:`apply_move` returns a new array.
``player`` is ``BLACK`` (+1) or ``WHITE`` (-1). A move is a ``(row, col)`` tuple or
``None`` for a pass.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .board import (
    BLACK,
    BOARD_SIZE,
    DIRECTIONS,
    EMPTY,
    WHITE,
    in_bounds,
    opponent,
)

Move = Optional[Tuple[int, int]]


def flips_for_move(board: np.ndarray, player: int, move: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Return the list of opponent discs that would be flipped by ``player``
    playing at ``move``.

    Returns an empty list if the move is illegal (off board, occupied square, or
    flips nothing).
    """
    row, col = move
    if not in_bounds(row, col) or board[row, col] != EMPTY:
        return []
    opp = opponent(player)
    flipped: List[Tuple[int, int]] = []
    for drow, dcol in DIRECTIONS:
        r, c = row + drow, col + dcol
        line: List[Tuple[int, int]] = []
        while in_bounds(r, c) and board[r, c] == opp:
            line.append((r, c))
            r += drow
            c += dcol
        if line and in_bounds(r, c) and board[r, c] == player:
            flipped.extend(line)
    return flipped


def is_legal_move(board: np.ndarray, player: int, move: Move) -> bool:
    """``None`` (pass) is legal only when ``player`` has no placing move."""
    if move is None:
        return not has_any_move(board, player)
    return len(flips_for_move(board, player, move)) > 0


def legal_moves(board: np.ndarray, player: int) -> List[Tuple[int, int]]:
    """All legal placing moves for ``player``, sorted (row, then col)."""
    moves: List[Tuple[int, int]] = []
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board[row, col] != EMPTY:
                continue
            if flips_for_move(board, player, (row, col)):
                moves.append((row, col))
    return moves


def has_any_move(board: np.ndarray, player: int) -> bool:
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board[row, col] == EMPTY and flips_for_move(board, player, (row, col)):
                return True
    return False


def apply_move(board: np.ndarray, player: int, move: Move) -> np.ndarray:
    """Return a new board after ``player`` plays ``move``.

    Raises ``ValueError`` for an illegal move (including a pass when a legal
    placing move exists). Turn switching is *not* handled here — see
    :func:`next_player`.
    """
    if move is None:
        if has_any_move(board, player):
            raise ValueError("illegal pass: player has at least one legal move")
        return board.copy()

    flipped = flips_for_move(board, player, move)
    if not flipped:
        raise ValueError(f"illegal move {move} for player {player}")
    new_board = board.copy()
    row, col = move
    new_board[row, col] = player
    for r, c in flipped:
        new_board[r, c] = player
    return new_board


def next_player(board: np.ndarray, player: int) -> Optional[int]:
    """Whose turn it is after ``player`` moves.

    Returns the opponent if they have a move, else ``player`` again if *they*
    still have a move (opponent forced to pass), else ``None`` (game over).
    """
    opp = opponent(player)
    if has_any_move(board, opp):
        return opp
    if has_any_move(board, player):
        return player
    return None


def is_terminal(board: np.ndarray) -> bool:
    """True when neither player has a legal move (or the board is full)."""
    return not has_any_move(board, BLACK) and not has_any_move(board, WHITE)


def score(board: np.ndarray) -> Tuple[int, int]:
    """``(black_discs, white_discs)``."""
    return int(np.sum(board == BLACK)), int(np.sum(board == WHITE))


def winner(board: np.ndarray) -> int:
    """``BLACK``, ``WHITE``, or ``0`` for a draw, by disc count.

    Meaningful once the game is terminal, but callable at any time.
    """
    black, white = score(board)
    if black > white:
        return BLACK
    if white > black:
        return WHITE
    return 0

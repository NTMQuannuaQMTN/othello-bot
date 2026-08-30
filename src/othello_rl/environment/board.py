"""Board representation, constants and coordinate helpers for 8x8 Othello.

Representation
-------------
- The board is a ``numpy.ndarray`` of shape ``(8, 8)`` and dtype ``int8``.
- Cell values: ``+1`` = Black, ``-1`` = White, ``0`` = empty.
- ``board[row, col]``; ``row == 0`` is the top rank, ``col == 0`` the left file.
- Human notation ``"d3"``: file ``a``..``h`` -> ``col`` 0..7, rank ``1``..``8`` ->
  ``row`` 0..7. So ``"d3"`` == ``(row=2, col=3)``.
- Standard start: ``(3,3)=White``, ``(3,4)=Black``, ``(4,3)=Black``, ``(4,4)=White``.
- Black moves first.

Moves are ``(row, col)`` tuples. ``None`` means "pass". The flat action index is
``row * 8 + col`` (0..63); ``64`` is the pass action.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

BOARD_SIZE = 8
NUM_SQUARES = BOARD_SIZE * BOARD_SIZE

EMPTY = 0
BLACK = 1
WHITE = -1

PASS_ACTION = NUM_SQUARES  # 64

Move = Optional[Tuple[int, int]]

# The eight directions: N, NE, E, SE, S, SW, W, NW (as (drow, dcol)).
DIRECTIONS: Tuple[Tuple[int, int], ...] = (
    (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1),
)

_FILES = "abcdefgh"


def opponent(player: int) -> int:
    """Return the other player (`BLACK` <-> `WHITE`)."""
    if player not in (BLACK, WHITE):
        raise ValueError(f"player must be BLACK(1) or WHITE(-1), got {player!r}")
    return -player


def initial_board() -> np.ndarray:
    """Return a fresh board with the standard four central discs."""
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    board[3, 3] = WHITE
    board[3, 4] = BLACK
    board[4, 3] = BLACK
    board[4, 4] = WHITE
    return board


def in_bounds(row: int, col: int) -> bool:
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE


def rc_to_action(row: int, col: int) -> int:
    if not in_bounds(row, col):
        raise ValueError(f"({row}, {col}) is off the board")
    return row * BOARD_SIZE + col


def action_to_rc(action: int) -> Move:
    """Convert a flat action index to a move. ``PASS_ACTION`` -> ``None``."""
    if action == PASS_ACTION:
        return None
    if not 0 <= action < NUM_SQUARES:
        raise ValueError(f"action out of range: {action}")
    return divmod(action, BOARD_SIZE)


def parse_square(name: str) -> Tuple[int, int]:
    """Parse ``"d3"`` (file+rank) into ``(row, col)``."""
    s = name.strip().lower()
    if len(s) != 2 or s[0] not in _FILES or not s[1].isdigit():
        raise ValueError(f"bad square name: {name!r}")
    col = _FILES.index(s[0])
    rank = int(s[1])
    if not 1 <= rank <= BOARD_SIZE:
        raise ValueError(f"bad rank in square name: {name!r}")
    row = rank - 1
    return row, col


def square_name(move: Tuple[int, int]) -> str:
    """Inverse of :func:`parse_square`."""
    row, col = move
    if not in_bounds(row, col):
        raise ValueError(f"({row}, {col}) is off the board")
    return f"{_FILES[col]}{row + 1}"


def render_board(board: np.ndarray, marks: Optional[List[Tuple[int, int]]] = None) -> str:
    """ASCII rendering. ``marks`` cells are shown as ``*`` when empty."""
    marks_set = set(marks or [])
    glyph = {BLACK: "X", WHITE: "O", EMPTY: "."}
    lines = ["  " + " ".join(_FILES)]
    for r in range(BOARD_SIZE):
        cells = []
        for c in range(BOARD_SIZE):
            v = int(board[r, c])
            if v == EMPTY and (r, c) in marks_set:
                cells.append("*")
            else:
                cells.append(glyph[v])
        lines.append(f"{r + 1} " + " ".join(cells))
    return "\n".join(lines)


# ``rules`` is imported after all names above are defined to avoid a circular
# import (``rules`` imports names from this module).
from . import rules  # noqa: E402


class Board:
    """Immutable-friendly Othello game state: a board array + side to move.

    Every mutating operation returns a *new* ``Board``; the wrapped array is never
    modified in place by this class.
    """

    __slots__ = ("array", "player")

    def __init__(self, array: np.ndarray, player: int, _own: bool = False):
        arr = np.asarray(array)
        if arr.shape != (BOARD_SIZE, BOARD_SIZE):
            raise ValueError(f"board array must be {BOARD_SIZE}x{BOARD_SIZE}")
        if player not in (BLACK, WHITE):
            raise ValueError("player must be BLACK(1) or WHITE(-1)")
        # Take an owned copy unless the caller guarantees a fresh, dedicated array
        # (``_own=True``, used only by internal transitions). This keeps a Board
        # truly immutable and stops it from aliasing/freezing a caller's array.
        if _own and arr.dtype == np.int8:
            self.array = arr
        else:
            self.array = np.array(arr, dtype=np.int8)
        self.array.flags.writeable = False
        self.player = int(player)

    # -- construction -------------------------------------------------------
    @classmethod
    def initial(cls) -> "Board":
        return cls(initial_board(), BLACK, _own=True)

    def copy(self) -> "Board":
        return Board(self.array.copy(), self.player, _own=True)

    # -- queries ----------------------------------------------------------
    def legal_moves(self) -> List[Tuple[int, int]]:
        return rules.legal_moves(self.array, self.player)

    def must_pass(self) -> bool:
        """True when the side to move has no placing move but the game is live."""
        return not self.is_terminal() and not rules.has_any_move(self.array, self.player)

    def legal_actions(self) -> List[int]:
        """Flat action indices. Returns ``[PASS_ACTION]`` when a pass is forced."""
        moves = self.legal_moves()
        if moves:
            return [rc_to_action(r, c) for (r, c) in moves]
        if not self.is_terminal():
            return [PASS_ACTION]
        return []

    def is_legal(self, move) -> bool:
        move = self._coerce(move)
        return rules.is_legal_move(self.array, self.player, move)

    def is_terminal(self) -> bool:
        return rules.is_terminal(self.array)

    def scores(self) -> Tuple[int, int]:
        return rules.score(self.array)

    def winner(self) -> int:
        return rules.winner(self.array)

    # -- transitions -----------------------------------------------------
    def _coerce(self, move) -> Move:
        if move is None:
            return None
        if isinstance(move, (int, np.integer)):
            return action_to_rc(int(move))
        return (int(move[0]), int(move[1]))

    def apply(self, move) -> "Board":
        """Play ``move`` (``(row,col)``, flat int, or ``None`` for pass) and
        switch to whichever player moves next. Raises on an illegal move."""
        move = self._coerce(move)
        new_array = rules.apply_move(self.array, self.player, move)  # fresh array
        nxt = rules.next_player(new_array, self.player)
        if nxt is None:  # terminal: keep a valid player value for the state
            nxt = opponent(self.player)
        return Board(new_array, nxt, _own=True)

    # -- misc -----------------------------------------------------------
    def render(self) -> str:
        return render_board(self.array, self.legal_moves())

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, Board)
            and self.player == other.player
            and np.array_equal(self.array, other.array)
        )

    def __hash__(self) -> int:
        return hash((self.player, self.array.tobytes()))

    def __repr__(self) -> str:
        b, w = self.scores()
        side = "BLACK" if self.player == BLACK else "WHITE"
        return f"<Board to_move={side} X={b} O={w}>"

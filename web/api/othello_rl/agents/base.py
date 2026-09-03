"""Common agent interface.

An agent receives an immutable :class:`~othello_rl.environment.board.Board` (which
carries the side to move) and returns a legal move: a ``(row, col)`` tuple, or
``None`` when the side to move is forced to pass.
"""
from __future__ import annotations

import abc
from typing import Optional, Tuple

from othello_rl.environment.board import Board

Move = Optional[Tuple[int, int]]


class Agent(abc.ABC):
    """Base class for all move-selecting agents."""

    #: Human-readable identifier, used in tournament tables and logs.
    name: str = "agent"

    def reset(self) -> None:
        """Hook called at the start of each game. Override if stateful."""

    @abc.abstractmethod
    def select_move(self, state: Board) -> Move:
        """Return a legal move for ``state`` (or ``None`` to pass)."""

    def __call__(self, state: Board) -> Move:
        return self.select_move(state)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"


def forced_pass(state: Board) -> bool:
    """True when the only legal action is to pass."""
    return state.must_pass()

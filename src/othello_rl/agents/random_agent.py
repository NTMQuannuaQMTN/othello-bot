"""Uniformly-random legal-move agent."""
from __future__ import annotations

import random
from typing import Optional

from othello_rl.environment.board import Board
from .base import Agent, Move


class RandomAgent(Agent):
    name = "random"

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)
        self._seed = seed

    def select_move(self, state: Board) -> Move:
        moves = state.legal_moves()
        if not moves:
            return None
        return self._rng.choice(moves)

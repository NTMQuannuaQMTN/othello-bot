"""Greedy agent: play the move that flips the most opponent discs right now.

Deterministic: ties are broken by the smallest flat action index
(``row * 8 + col``), i.e. top-left-most square.
"""
from __future__ import annotations

from othello_rl.environment import rules
from othello_rl.environment.board import Board
from .base import Agent, Move


class GreedyAgent(Agent):
    name = "greedy"

    def select_move(self, state: Board) -> Move:
        moves = state.legal_moves()
        if not moves:
            return None
        best = max(
            moves,
            key=lambda m: (len(rules.flips_for_move(state.array, state.player, m)),
                           -(m[0] * 8 + m[1])),
        )
        return best

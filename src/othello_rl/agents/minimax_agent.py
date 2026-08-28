"""Minimax agent with alpha-beta pruning and a heuristic leaf evaluation.

Implemented in negamax style: every node returns the position value **from the
perspective of the side to move at that node**.

Othello allows a player to move again when the opponent has no legal move, so
strict sign-alternation does not hold. Each child is handled explicitly:

- child's side to move == this node's side to move  (opponent was forced to pass)
  -> add the child value directly, window unchanged
- otherwise (normal alternation)
  -> add the negated child value, window negated/swapped

A node with no legal moves but a non-terminal position performs a pass (which
does not consume search depth). Deterministic: children are examined in ascending
action-index order; ties keep the first (lowest-index) move.
"""
from __future__ import annotations

from typing import Dict, Optional

from othello_rl.environment.board import Board
from .base import Agent, Move
from .heuristic_agent import DEFAULT_WEIGHTS, evaluate

INF = float("inf")


class MinimaxAgent(Agent):
    def __init__(self, depth: int = 3, weights: Optional[Dict[str, float]] = None):
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.depth = depth
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}
        self.name = f"minimax{depth}"
        self.nodes = 0  # diagnostic: nodes visited on the last search

    # -- public -----------------------------------------------------------
    def select_move(self, state: Board) -> Move:
        moves = state.legal_moves()
        if not moves:
            return None
        self.nodes = 0
        best_move = min(moves, key=lambda mv: mv[0] * 8 + mv[1])
        best_val = -INF
        alpha = -INF
        for m in sorted(moves, key=lambda mv: mv[0] * 8 + mv[1]):
            val = self._child_value(state, m, self.depth, alpha, INF, prune=True)
            if val > best_val:
                best_val = val
                best_move = m
            alpha = max(alpha, best_val)
        return best_move

    def search_value(self, state: Board) -> float:
        """Full negamax value of ``state`` with no pruning — for tests."""
        return self._value(state, self.depth, -INF, INF, prune=False)

    # -- internal ---------------------------------------------------------
    def _child_value(self, state: Board, move: Move, depth: int,
                     alpha: float, beta: float, prune: bool) -> float:
        child = state.apply(move)
        if child.player == state.player:  # opponent forced to pass
            return self._value(child, depth - 1, alpha, beta, prune)
        return -self._value(child, depth - 1, -beta, -alpha, prune)

    def _value(self, state: Board, depth: int, alpha: float, beta: float,
               prune: bool = True) -> float:
        self.nodes += 1
        if state.is_terminal() or depth <= 0:
            return evaluate(state.array, state.player, self.weights)

        moves = state.legal_moves()
        if not moves:  # forced pass; opponent is guaranteed a move here
            return -self._value(state.apply(None), depth, -beta, -alpha, prune)

        value = -INF
        for m in sorted(moves, key=lambda mv: mv[0] * 8 + mv[1]):
            value = max(value, self._child_value(state, m, depth, alpha, beta, prune))
            alpha = max(alpha, value)
            if prune and alpha >= beta:
                break
        return value

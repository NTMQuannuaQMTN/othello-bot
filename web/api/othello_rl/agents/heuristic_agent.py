"""Configurable static-evaluation heuristic agent.

The position value is a weighted sum of interpretable features, always computed
**from the perspective of a given player** (positive = good for that player):

- ``disc_diff``      : (my discs - opponent discs), normalised to [-1, 1]
- ``mobility``       : (my legal moves - opp legal moves) / (sum + 1)
- ``corners``        : (my corners - opp corners) / 4
- ``corner_danger``  : (my X/C-squares next to an *empty* corner
                        - opp's) / 8   -- weight is normally negative
- ``edges``          : (my edge discs - opp edge discs) / 24  (corners excluded)

(A disc/edge "stability" and endgame "parity" term would need turn/parity
information that the static board array does not carry; deferred.)

The agent chooses the move leading to the position with the best value for the
side to move (1-ply lookahead). Deterministic tie-break: smallest action index.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from othello_rl.environment import rules
from othello_rl.environment.board import BLACK, WHITE, Board, opponent
from .base import Agent, Move

CORNERS = ((0, 0), (0, 7), (7, 0), (7, 7))
# X/C squares adjacent to each corner
_ADJ = {
    (0, 0): ((0, 1), (1, 0), (1, 1)),
    (0, 7): ((0, 6), (1, 7), (1, 6)),
    (7, 0): ((7, 1), (6, 0), (6, 1)),
    (7, 7): ((7, 6), (6, 7), (6, 6)),
}

DEFAULT_WEIGHTS: Dict[str, float] = {
    "disc_diff": 1.0,
    "mobility": 8.0,
    "corners": 25.0,
    "corner_danger": -8.0,
    "edges": 4.0,
}


def _count(board: np.ndarray, player: int) -> int:
    return int(np.sum(board == player))


def evaluate(board: np.ndarray, player: int, weights: Optional[Dict[str, float]] = None) -> float:
    """Static value of ``board`` from ``player``'s perspective."""
    w = DEFAULT_WEIGHTS if weights is None else {**DEFAULT_WEIGHTS, **weights}
    opp = opponent(player)

    my_d, opp_d = _count(board, player), _count(board, opp)
    empties = 64 - my_d - opp_d

    if empties == 0 or (not rules.has_any_move(board, BLACK) and not rules.has_any_move(board, WHITE)):
        # terminal: decisive value dominates everything else
        if my_d > opp_d:
            return 1e6 + (my_d - opp_d)
        if my_d < opp_d:
            return -1e6 - (opp_d - my_d)
        return 0.0

    disc_diff = (my_d - opp_d) / max(1, my_d + opp_d)

    my_mob = len(rules.legal_moves(board, player))
    opp_mob = len(rules.legal_moves(board, opp))
    mobility = (my_mob - opp_mob) / (my_mob + opp_mob + 1)

    my_c = sum(1 for rc in CORNERS if board[rc] == player)
    opp_c = sum(1 for rc in CORNERS if board[rc] == opp)
    corners = (my_c - opp_c) / 4.0

    my_x = opp_x = 0
    for corner, adj in _ADJ.items():
        if board[corner] != 0:
            continue  # only dangerous while the corner is still empty
        for rc in adj:
            if board[rc] == player:
                my_x += 1
            elif board[rc] == opp:
                opp_x += 1
    corner_danger = (my_x - opp_x) / 8.0

    edge_cells = [(0, c) for c in range(1, 7)] + [(7, c) for c in range(1, 7)] + \
                 [(r, 0) for r in range(1, 7)] + [(r, 7) for r in range(1, 7)]
    my_e = sum(1 for rc in edge_cells if board[rc] == player)
    opp_e = sum(1 for rc in edge_cells if board[rc] == opp)
    edges = (my_e - opp_e) / 24.0

    return (
        w["disc_diff"] * disc_diff
        + w["mobility"] * mobility
        + w["corners"] * corners
        + w["corner_danger"] * corner_danger
        + w["edges"] * edges
    )


class HeuristicAgent(Agent):
    name = "heuristic"

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}

    def select_move(self, state: Board) -> Move:
        moves = state.legal_moves()
        if not moves:
            return None
        best_move = None
        best_val = -float("inf")
        for m in sorted(moves, key=lambda mv: mv[0] * 8 + mv[1]):
            child = state.apply(m)
            # child.player is the mover's opponent (or same on forced pass);
            # evaluate from the current mover's perspective.
            val = evaluate(child.array, state.player, self.weights)
            if val > best_val:
                best_val = val
                best_move = m
        return best_move

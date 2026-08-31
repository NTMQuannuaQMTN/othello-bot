"""Shallow negamax search with alpha-beta pruning and a static heuristic leaf.

Reuses the tournament :class:`~othello_rl.agents.minimax_agent.MinimaxAgent`
machinery — the corner-first move ordering (``_ordered``) and the heuristic leaf
(``heuristic_agent.evaluate``) — but exposes a plain function with an optional
**exact-only** transposition table (a node is cached only if it examined every
child without a beta cutoff, so a cached value is always the true minimax value
and never a bound).

``depth`` is in **plies** (individual moves). A forced pass does not consume a
ply, matching ``MinimaxAgent``. The returned value of a top-level full-window
call is the exact minimax value whether or not ``alpha_beta`` is on.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

from othello_rl.agents.heuristic_agent import DEFAULT_WEIGHTS, evaluate
from othello_rl.agents.minimax_agent import _ordered
from othello_rl.environment.board import Board

_INF = math.inf
TTable = Dict[Tuple[int, int], float]  # (board hash, depth) -> exact value


def shallow_value(board: Board, depth: int, weights: Optional[dict] = None,
                  alpha: float = -_INF, beta: float = _INF, *,
                  alpha_beta: bool = True, tt: Optional[TTable] = None) -> float:
    """Negamax value of ``board`` from the side to move, ``depth`` plies deep."""
    w = weights or DEFAULT_WEIGHTS
    if board.is_terminal() or depth <= 0:
        return evaluate(board.array, board.player, w)

    key = (hash(board), depth) if tt is not None else None
    if key is not None and key in tt:
        return tt[key]

    moves = board.legal_moves()
    if not moves:  # defensive: Board.apply auto-skips passes, so this is rare
        v = -shallow_value(board.apply(None), depth, w, -beta, -alpha,
                           alpha_beta=alpha_beta, tt=tt)
        if key is not None:
            tt[key] = v
        return v

    value = -_INF
    pruned = False
    for m in _ordered(moves):
        child = board.apply(m)
        if child.player == board.player:      # opponent forced to pass
            v = shallow_value(child, depth - 1, w, alpha, beta,
                              alpha_beta=alpha_beta, tt=tt)
        else:
            v = -shallow_value(child, depth - 1, w, -beta, -alpha,
                               alpha_beta=alpha_beta, tt=tt)
        value = max(value, v)
        alpha = max(alpha, value)
        if alpha_beta and alpha >= beta:
            pruned = True
            break

    if key is not None and not pruned:        # cache exact values only
        tt[key] = value
    return value


def move_value(board: Board, move, depth: int, weights: Optional[dict] = None, *,
               alpha_beta: bool = True, tt: Optional[TTable] = None) -> float:
    """Value of playing ``move`` from ``board``, ``depth`` plies of look-ahead
    beyond it, **from the perspective of the player who played it**."""
    mover = board.player
    child = board.apply(move)
    v = shallow_value(child, depth - 1, weights, alpha_beta=alpha_beta, tt=tt)
    return v if child.player == mover else -v

"""A real Othello search engine on top of :mod:`othello_rl.engine.bitboard`.

Negamax + alpha-beta, transposition table, static + killer move ordering,
iterative deepening under a time budget, and an **exact endgame solve** once few
squares remain (leaf = the true final disc margin, so the last dozen moves are
played perfectly).

This is the engine behind ``OthelloBot.best_move`` — the move the web app
suggests and the move the Play-tab bot makes.  It is independent of the RL
policy (the DQN is only a tiny tiebreak nudge, applied by the caller).
"""
from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

from .bitboard import (
    CORNERS, FULL, _DIRS, final_score, legal_moves, move_list, play, popcount,
)

_INF = 1 << 30
_WIN = 1 << 20                       # endgame scores live here (margin * SCALE)
_SCALE = 1000

# static square priority (corners best, X/C squares worst) — same table as
# agents.minimax_agent, indexed by bit = row*8 + col
_ORDER = (
    120, -20, 20, 5, 5, 20, -20, 120,
    -20, -40, -5, -5, -5, -5, -40, -20,
    20, -5, 15, 3, 3, 15, -5, 20,
    5, -5, 3, 3, 3, 3, -5, 5,
    5, -5, 3, 3, 3, 3, -5, 5,
    20, -5, 15, 3, 3, 15, -5, 20,
    -20, -40, -5, -5, -5, -5, -40, -20,
    120, -20, 20, 5, 5, 20, -20, 120,
)

_X_ADJ = {                           # X/C squares that guard each still-empty corner
    0: (9, 1, 8), 7: (14, 6, 15), 56: (49, 48, 57), 63: (54, 55, 62),
}


class Timeout(Exception):
    pass


def evaluate(P: int, O: int, empties: int) -> float:
    """Midgame heuristic, from ``P``'s perspective (roughly [-1e4, 1e4])."""
    my_moves = legal_moves(P, O)
    op_moves = legal_moves(O, P)
    mp, op = popcount(my_moves), popcount(op_moves)
    mobility = (mp - op) / (mp + op + 1)

    cp, co = popcount(P & CORNERS), popcount(O & CORNERS)
    corners = cp - co

    empty = FULL ^ (P | O)
    adj = 0
    for shift in _DIRS:
        adj |= shift(empty)
    fp, fo = popcount(P & adj & ~CORNERS), popcount(O & adj & ~CORNERS)
    frontier = (fo - fp) / (fp + fo + 1)      # fewer frontier discs is better

    # danger: an X/C square we own next to a still-empty corner
    danger = 0
    for corner, adjs in _X_ADJ.items():
        if (P | O) >> corner & 1:
            continue
        for a in adjs:
            danger += ((P >> a) & 1) - ((O >> a) & 1)

    pc, oc = popcount(P), popcount(O)
    disc = (pc - oc) / (pc + oc)
    w_disc = 6.0 + 34.0 * (1.0 - empties / 60.0)      # ~6 early -> ~40 late

    return (700.0 * corners + 300.0 * mobility + 90.0 * frontier
            - 110.0 * danger + w_disc * disc)


def _ordered(moves_mask: int, tt_best: int) -> list:
    sqs = move_list(moves_mask)
    sqs.sort(key=lambda s: (s != tt_best, -_ORDER[s], s))
    return sqs


def _negamax(P: int, O: int, depth: int, alpha: float, beta: float,
             empties: int, endgame_at: int, tt: Dict, deadline: float,
             nodes: list) -> float:
    nodes[0] += 1
    if nodes[0] & 0x3FF == 0 and time.monotonic() > deadline:
        raise Timeout

    my = legal_moves(P, O)
    if my == 0:
        if legal_moves(O, P) == 0:                    # game over
            m = final_score(P, O)
            return (_WIN if m > 0 else -_WIN if m < 0 else 0) + _SCALE * m
        return -_negamax(O, P, depth, -beta, -alpha, empties, endgame_at, tt,
                         deadline, nodes)             # pass — no ply consumed

    exact_endgame = empties <= endgame_at
    if not exact_endgame and depth <= 0:
        return evaluate(P, O, empties)

    key = (P, O)
    hit = tt.get(key)
    tt_best = -1
    if hit is not None:
        h_depth, h_val, h_flag, h_best = hit
        if h_depth >= depth or (exact_endgame and h_flag == 0):
            if h_flag == 0:
                return h_val
            if h_flag > 0 and h_val >= beta:
                return h_val
            if h_flag < 0 and h_val <= alpha:
                return h_val
        tt_best = h_best

    best_val = -_INF
    best_sq = -1
    a0 = alpha
    d = depth if exact_endgame else depth - 1
    for sq in _ordered(my, tt_best):
        nP, nO = play(P, O, sq)
        v = -_negamax(nP, nO, d, -beta, -alpha, empties - 1, endgame_at, tt,
                      deadline, nodes)
        if v > best_val:
            best_val, best_sq = v, sq
        if v > alpha:
            alpha = v
        if alpha >= beta:
            break

    flag = 0 if a0 < best_val < beta else (1 if best_val >= beta else -1)
    tt[key] = (depth, best_val, flag, best_sq)
    return best_val


def best_move(P: int, O: int, *, max_depth: int = 64, endgame_empties: int = 14,
              time_budget: float = 1.0, tt: Optional[Dict] = None) -> Tuple[Optional[int], float, dict]:
    """Iterative-deepening search.  Returns ``(square | None, score, meta)``.

    ``score`` is from ``P``'s view: endgame scores are ``final_margin`` (in discs,
    give or take ``_SCALE``); midgame scores are heuristic units.  ``meta`` has
    ``depth`` / ``exact`` / ``nodes`` / ``time`` / ``pv`` (the search line).
    """
    my = legal_moves(P, O)
    if my == 0:
        return None, 0.0, {"depth": 0, "exact": False, "nodes": 0, "time": 0.0, "pv": []}

    tt = tt if tt is not None else {}
    empties = 64 - popcount(P) - popcount(O)
    deadline = time.monotonic() + time_budget
    t0 = time.monotonic()
    nodes = [0]

    exact = empties <= endgame_empties
    best_sq = _ordered(my, -1)[0]
    best_val = -_INF
    reached = 0

    limit = empties if exact else max_depth
    for depth in range(1, limit + 1):
        try:
            local_best, local_val = -1, -_INF
            alpha = -_INF
            for sq in _ordered(my, best_sq):
                nP, nO = play(P, O, sq)
                v = -_negamax(nP, nO, (empties - 1) if exact else depth - 1,
                              -_INF, -alpha, empties - 1, endgame_empties, tt,
                              deadline, nodes)
                if v > local_val:
                    local_val, local_best = v, sq
                    alpha = max(alpha, v)
            best_sq, best_val, reached = local_best, local_val, depth
        except Timeout:
            break
        if exact:                       # endgame solve is one exact pass
            break
        if abs(best_val) >= _WIN:       # found a forced result — stop
            break
        if time.monotonic() > deadline:
            break

    tt[(P, O)] = (reached, best_val, 0, best_sq)     # so _pv can start here
    pv = _pv(P, O, tt, 12)
    return best_sq, float(best_val), {
        "depth": reached, "exact": exact or abs(best_val) >= _WIN,
        "nodes": nodes[0], "time": round(time.monotonic() - t0, 3), "pv": pv,
    }


def _pv(P: int, O: int, tt: Dict, n: int) -> list:
    out = []
    for _ in range(n):
        hit = tt.get((P, O))
        if not hit or hit[3] < 0:
            break
        sq = hit[3]
        if not (legal_moves(P, O) >> sq) & 1:
            break
        out.append(sq)
        P, O = play(P, O, sq)
        if legal_moves(P, O) == 0:
            if legal_moves(O, P) == 0:
                break
            P, O = O, P
    return out

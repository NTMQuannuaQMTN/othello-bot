"""Fast bitboard Othello: move generation, flips, and board conversion.

Two ``int`` (64-bit) masks ``(P, O)`` — ``P`` = the side-to-move's discs, ``O`` =
the opponent's.  Bit ``i`` is square ``(row=i // 8, col=i % 8)`` — the same
``row * 8 + col`` encoding as :mod:`othello_rl.environment.board`, so bit 0 is
``a1`` and bit 63 is ``h8``.

Pure Python (no numpy) — this is the hot path for the search engine
(:mod:`othello_rl.engine.solver`).  It is a *separate* implementation from the
numpy ``environment`` engine and is property-tested against it.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

FULL = (1 << 64) - 1
_NOT_A = 0xFEFEFEFEFEFEFEFE      # every square except the a-file (col 0)
_NOT_H = 0x7F7F7F7F7F7F7F7F      # every square except the h-file (col 7)

CORNERS = (1 << 0) | (1 << 7) | (1 << 56) | (1 << 63)

#: (shift function) for each of the 8 directions
_DIRS = (
    lambda x: (x << 1) & _NOT_A & FULL,     # E   (col + 1)
    lambda x: (x >> 1) & _NOT_H,            # W   (col - 1)
    lambda x: (x << 8) & FULL,              # S   (row + 1)
    lambda x: x >> 8,                       # N   (row - 1)
    lambda x: (x << 9) & _NOT_A & FULL,     # SE
    lambda x: (x << 7) & _NOT_H & FULL,     # SW
    lambda x: (x >> 7) & _NOT_A,            # NE
    lambda x: (x >> 9) & _NOT_H,            # NW
)


def popcount(x: int) -> int:
    return bin(x).count("1")


def from_grid(array, player: int) -> Tuple[int, int]:
    """``(8, 8)`` int8 grid + side to move -> ``(P, O)`` bitboards."""
    a = np.asarray(array)
    p = o = 0
    for i in range(64):
        v = int(a[i // 8, i % 8])
        if v == player:
            p |= 1 << i
        elif v == -player:
            o |= 1 << i
    return p, o


def to_grid(P: int, O: int, player: int) -> np.ndarray:
    """``(P, O)`` + which colour ``P`` is -> ``(8, 8)`` int8 grid."""
    a = np.zeros((8, 8), dtype=np.int8)
    for i in range(64):
        if (P >> i) & 1:
            a[i // 8, i % 8] = player
        elif (O >> i) & 1:
            a[i // 8, i % 8] = -player
    return a


def legal_moves(P: int, O: int) -> int:
    """Bitmask of squares where ``P`` may play."""
    empty = FULL ^ (P | O)
    moves = 0
    for shift in _DIRS:
        run = shift(P) & O
        run |= shift(run) & O
        run |= shift(run) & O
        run |= shift(run) & O
        run |= shift(run) & O
        run |= shift(run) & O
        moves |= shift(run) & empty
    return moves


def flips(P: int, O: int, sq: int) -> int:
    """Bitmask of ``O`` discs flipped if ``P`` plays at ``sq`` (0 if illegal)."""
    m = 1 << sq
    out = 0
    for shift in _DIRS:
        line = 0
        x = shift(m) & O
        while x:
            line |= x
            x = shift(x) & O
        if shift(line) & P:            # the square just past the run is ours
            out |= line
    return out


def play(P: int, O: int, sq: int) -> Tuple[int, int]:
    """Play ``sq`` for ``P``; return ``(P, O)`` **from the opponent's view**
    (i.e. the returned ``P`` is the next side to move).  Assumes ``sq`` legal."""
    f = flips(P, O, sq)
    newP = P | f | (1 << sq)
    newO = O ^ f
    return newO, newP                  # swap: opponent moves next


def move_list(mask: int) -> List[int]:
    """Set-bit indices of ``mask``, ascending."""
    out = []
    while mask:
        low = mask & -mask
        out.append(low.bit_length() - 1)
        mask ^= low
    return out


def final_score(P: int, O: int) -> int:
    """Game-over disc margin for ``P`` (empties go to the leader, the standard
    tournament rule): ``+n`` P wins by n, ``-n`` O wins by n, ``0`` draw."""
    pc, oc = popcount(P), popcount(O)
    empties = 64 - pc - oc
    if pc > oc:
        return pc + empties - oc
    if oc > pc:
        return pc - (oc + empties)
    return 0

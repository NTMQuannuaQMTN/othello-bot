"""Potential-based reward shaping for corner safety.

The sparse ±1 DQN reward gives the small conv net almost no signal about corners,
so it happily plays an **X-square** (b2 / g2 / b7 / g7) next to a still-empty
corner and loses that corner a few plies later. This module adds a
*potential-based* shaping term (Ng, Harada & Russell, 1999):

    F(s, s') = γ · Φ(s') − Φ(s)

which provably leaves the optimal policy unchanged but turns the sparse problem
into a dense one: Φ rises when the learner holds corners and falls when it sits
on an X- or C-square guarding a still-empty corner.

Wire it in via ``FixedOpponentEnv(..., shaping=CornerShaping(...))`` — both the
curriculum (``scripts/train.py``) and self-play (``scripts/selfplay.py``) read a
``shaping:`` block from their config.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_CORNERS = ((0, 0), (0, 7), (7, 0), (7, 7))
#: the diagonal square guarding each corner (the classic "X-square")
_X_FOR_CORNER = {(0, 0): (1, 1), (0, 7): (1, 6), (7, 0): (6, 1), (7, 7): (6, 6)}
#: the two orthogonal squares guarding each corner (the "C-squares")
_C_FOR_CORNER = {
    (0, 0): ((0, 1), (1, 0)), (0, 7): ((0, 6), (1, 7)),
    (7, 0): ((7, 1), (6, 0)), (7, 7): ((7, 6), (6, 7)),
}


@dataclass
class CornerShaping:
    """Potential-based corner-safety shaping. Weights are in reward units; keep
    them well below 1 (the terminal win/loss reward) so shaping only guides."""

    corner_weight: float = 0.20     # per net corner held (mine − opponent's)
    x_square_weight: float = 0.12   # per net X-square sat on next to an EMPTY corner
    c_square_weight: float = 0.04   # ... C-square
    gamma: float = 0.99
    enabled: bool = True

    def potential(self, board: np.ndarray, learner: int) -> float:
        """Φ(board) from the learner's perspective (higher = safer corners)."""
        if not self.enabled:
            return 0.0
        opp = -learner
        phi = 0.0
        for corner in _CORNERS:
            v = int(board[corner])
            if v == learner:
                phi += self.corner_weight
            elif v == opp:
                phi -= self.corner_weight
            else:  # corner still empty -> its guard squares are liabilities
                xv = int(board[_X_FOR_CORNER[corner]])
                if xv == learner:
                    phi -= self.x_square_weight
                elif xv == opp:
                    phi += self.x_square_weight
                for c in _C_FOR_CORNER[corner]:
                    cv = int(board[c])
                    if cv == learner:
                        phi -= self.c_square_weight
                    elif cv == opp:
                        phi += self.c_square_weight
        return phi

    def delta(self, board_before: np.ndarray, board_after: np.ndarray,
              learner: int, *, done: bool) -> float:
        """The shaping reward F = γ·Φ(s') − Φ(s) to add for this transition.
        Φ(terminal) is taken as 0, as the potential-based formulation requires."""
        if not self.enabled:
            return 0.0
        phi_s = self.potential(board_before, learner)
        phi_s2 = 0.0 if done else self.potential(board_after, learner)
        return self.gamma * phi_s2 - phi_s

    @classmethod
    def from_config(cls, cfg, gamma: float = 0.99) -> "CornerShaping":
        """Build from a plain dict (a ``shaping:`` config block) or None."""
        if not cfg:
            return cls(enabled=False)
        d = dict(cfg)
        d.setdefault("gamma", gamma)
        d.setdefault("enabled", True)
        allowed = {"corner_weight", "x_square_weight", "c_square_weight",
                   "gamma", "enabled"}
        return cls(**{k: v for k, v in d.items() if k in allowed})

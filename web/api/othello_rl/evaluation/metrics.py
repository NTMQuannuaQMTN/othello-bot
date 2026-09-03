"""Aggregate statistics for evaluation matches."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple


def win_rate(wins: float, n: int) -> float:
    return wins / n if n else 0.0


def wilson_interval(successes: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion.

    ``successes`` may be fractional (draws counted as 0.5). Returns ``(lo, hi)``
    clamped to ``[0, 1]``. For ``n == 0`` returns ``(0.0, 1.0)``.
    """
    if n <= 0:
        return 0.0, 1.0
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / denom
    return max(0.0, center - half), min(1.0, center + half)


@dataclass
class MatchSummary:
    name_a: str
    name_b: str
    num_games: int
    a_wins: int
    b_wins: int
    draws: int
    a_win_rate: float
    ci_low: float
    ci_high: float
    mean_disc_diff: float

    def significant_advantage(self) -> bool:
        """True when the 95% CI for A's score rate excludes 0.5."""
        return self.ci_low > 0.5 or self.ci_high < 0.5

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def summarize_match(match) -> MatchSummary:
    """Build a :class:`MatchSummary` from a
    :class:`~othello_rl.evaluation.tournament.MatchResult`."""
    lo, hi = wilson_interval(match.a_score, match.num_games)
    return MatchSummary(
        name_a=match.name_a,
        name_b=match.name_b,
        num_games=match.num_games,
        a_wins=match.a_wins,
        b_wins=match.b_wins,
        draws=match.draws,
        a_win_rate=match.a_win_rate,
        ci_low=lo,
        ci_high=hi,
        mean_disc_diff=match.mean_disc_diff,
    )

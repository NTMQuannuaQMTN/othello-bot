"""Internal Elo rating estimation.

IMPORTANT
---------
These ratings are **internal experimental Elo**: they are only comparable within
this project's own pool of agents and evaluation settings. They are NOT calibrated
to, and must never be reported as, any external online platform rating (e.g. an
Othello server / OGS / tournament rating). The constant :data:`KIND` marks this.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

KIND = "internal"
DEFAULT_RATING = 1500.0
_SCALE = 400.0


def expected_score(rating_a: float, rating_b: float) -> float:
    """Logistic expectation that A scores against B (in ``[0, 1]``)."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / _SCALE))


def update_pair(rating_a: float, rating_b: float, score_a: float, k: float = 32.0
                ) -> Tuple[float, float]:
    """One Elo update for a game where A scored ``score_a`` (1/0.5/0)."""
    ea = expected_score(rating_a, rating_b)
    delta = k * (score_a - ea)
    return rating_a + delta, rating_b - delta


@dataclass
class EloModel:
    """Iterative Elo fit over a list of (A, B, score_a) game records."""

    k: float = 24.0
    anchor: str = None  # optional agent whose rating is pinned to DEFAULT_RATING
    ratings: Dict[str, float] = field(default_factory=dict)
    kind: str = KIND

    def rating(self, name: str) -> float:
        return self.ratings.get(name, DEFAULT_RATING)

    def record_game(self, a: str, b: str, score_a: float) -> None:
        ra, rb = self.rating(a), self.rating(b)
        ra, rb = update_pair(ra, rb, score_a, self.k)
        self.ratings[a], self.ratings[b] = ra, rb

    def fit(self, games: Iterable[Tuple[str, str, float]], passes: int = 20,
            seed: int = 0) -> "EloModel":
        """Repeatedly sweep the game list (shuffled) to converge ratings."""
        games = list(games)
        rng = random.Random(seed)
        for _ in range(max(1, passes)):
            rng.shuffle(games)
            for a, b, score_a in games:
                self.record_game(a, b, score_a)
                if self.anchor is not None:
                    shift = DEFAULT_RATING - self.ratings.get(self.anchor, DEFAULT_RATING)
                    for name in self.ratings:
                        self.ratings[name] += shift
        return self

    def leaderboard(self) -> List[Tuple[str, float]]:
        return sorted(self.ratings.items(), key=lambda kv: kv[1], reverse=True)


def ratings_from_matches(matches, k: float = 24.0, passes: int = 25,
                         anchor: str = None, seed: int = 0) -> EloModel:
    """Fit an :class:`EloModel` from
    :class:`~othello_rl.evaluation.tournament.MatchResult` objects.

    Each individual game in every match contributes one record.
    """
    from othello_rl.environment.board import BLACK  # local import to avoid cycle

    records: List[Tuple[str, str, float]] = []
    for m in matches:
        # Reconstruct per-game A-scores from stored GameResults + alternation.
        for g_idx, gr in enumerate(m.games):
            a_is_black = (g_idx % 2 == 0)
            if gr.winner == 0:
                score_a = 0.5
            elif (gr.winner == BLACK) == a_is_black:
                score_a = 1.0
            else:
                score_a = 0.0
            records.append((m.name_a, m.name_b, score_a))
    model = EloModel(k=k, anchor=anchor)
    return model.fit(records, passes=passes, seed=seed)

"""Evaluation: tournaments, aggregate metrics, internal Elo, reports."""
from .tournament import (
    GameResult,
    MatchResult,
    play_game,
    play_match,
    round_robin,
)
from .metrics import MatchSummary, summarize_match, wilson_interval, win_rate
from .elo import EloModel, expected_score, ratings_from_matches, update_pair
from .report import generate_report
from .harness import evaluate_agent, flatten_eval

__all__ = [
    "GameResult",
    "MatchResult",
    "play_game",
    "play_match",
    "round_robin",
    "MatchSummary",
    "summarize_match",
    "wilson_interval",
    "win_rate",
    "EloModel",
    "expected_score",
    "update_pair",
    "ratings_from_matches",
    "generate_report",
    "evaluate_agent",
    "flatten_eval",
]

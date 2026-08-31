"""Bridges to *external* Othello engines, for evaluation only.

Nothing in here trains, fine-tunes or writes to ``checkpoints/`` / ``models/``.
The only external engine wired up today is **Egaroucid for Console** (GTP).
"""
from .egaroucid import EgaroucidEngine, EgaroucidError, find_egaroucid
from .match import (
    GameRecord,
    MatchSummary,
    finetune_on_records,
    play_game,
    records_to_training_games,
    run_match,
)

__all__ = [
    "EgaroucidEngine",
    "EgaroucidError",
    "find_egaroucid",
    "GameRecord",
    "MatchSummary",
    "play_game",
    "run_match",
    "finetune_on_records",
    "records_to_training_games",
]

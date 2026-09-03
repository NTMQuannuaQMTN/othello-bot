"""Web app: play against the bot and run Lichess-style move-by-move analysis.
Inference only — training is done offline (``scripts/train_*.py``)."""
from .bot_service import MoveAnalysis, OthelloBot
from .session import GameSession

__all__ = ["OthelloBot", "GameSession", "MoveAnalysis"]

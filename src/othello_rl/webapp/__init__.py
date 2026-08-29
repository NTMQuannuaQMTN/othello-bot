"""Web app: play against the bot, fine-tune it from finished games, and run
Lichess-style move-by-move analysis."""
from .bot_service import FineTuneConfig, MoveAnalysis, OthelloBot
from .session import GameSession

__all__ = ["OthelloBot", "GameSession", "FineTuneConfig", "MoveAnalysis"]

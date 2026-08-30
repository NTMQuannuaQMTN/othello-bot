"""In-memory single game between one human and the bot (local single-user tool)."""
from __future__ import annotations

from typing import List, Optional

from othello_rl.environment.board import (
    BLACK,
    WHITE,
    Board,
    PASS_ACTION,
    action_to_rc,
    square_name,
)
from .bot_service import OthelloBot, _san, _side


class GameSession:
    def __init__(self, bot: OthelloBot):
        self.bot = bot
        self.board = Board.initial()
        self.history: List[int] = []
        self.human_color = BLACK
        self.level = 0
        self.last_bot_moves: List[int] = []

    # -- lifecycle ----------------------------------------------------
    def new_game(self, human_color: str = "black", level: int = 0) -> dict:
        import random
        hc = human_color.lower()
        if hc.startswith("r"):
            hc = random.choice(["black", "white"])
        self.human_color = BLACK if hc.startswith("b") else WHITE
        self.level = int(level)
        self.board = Board.initial()
        self.history = []
        self.last_bot_moves = []
        if self.board.player != self.human_color:
            self._bot_turn()
        return self.state()

    # -- moves ------------------------------------------------------
    def human_move(self, action: int, bot_reply: bool = True) -> dict:
        if self.board.is_terminal():
            raise ValueError("game is over")
        if self.board.player != self.human_color:
            raise ValueError("not your turn")
        action = int(action)
        legal = _legal_actions(self.board)
        if action not in legal:
            raise ValueError(f"illegal move; legal = {legal}")
        self._apply(action)
        self.last_bot_moves = []
        if bot_reply:
            self._bot_turn()
        return self.state()

    def bot_move(self) -> dict:
        """Advance the bot if it is to move (used when the bot plays first, or a
        bot-vs-bot demo)."""
        if not self.board.is_terminal() and self.board.player != self.human_color:
            self._bot_turn()
        return self.state()

    # -- internals -------------------------------------------------
    def _apply(self, action: int) -> None:
        move = None if action == PASS_ACTION else action_to_rc(action)
        self.board = self.board.apply(move)
        self.history.append(action)

    def _bot_turn(self) -> None:
        self.last_bot_moves = []
        while not self.board.is_terminal() and self.board.player != self.human_color:
            if not self.board.legal_moves():
                self._apply(PASS_ACTION)
                self.last_bot_moves.append(PASS_ACTION)
                continue
            a = self.bot.select_action(self.board)
            self._apply(a)
            self.last_bot_moves.append(a)

    def _grid(self, b: Board) -> List[List[int]]:
        return [[int(b.array[r, c]) for c in range(8)] for r in range(8)]

    def _replay(self):
        """Per-move log + the board grid after every ply (index 0 = start)."""
        log: List[dict] = []
        b = Board.initial()
        grids = [self._grid(b)]
        for i, a in enumerate(self.history):
            log.append({
                "n": i + 1,
                "san": _san(a),
                "side": _side(b.player),
                "by": "you" if b.player == self.human_color else "bot",
                "pass": a == PASS_ACTION,
            })
            b = b.apply(None if a == PASS_ACTION else action_to_rc(a))
            grids.append(self._grid(b))
        return log, grids

    # -- serialisation --------------------------------------------
    def state(self) -> dict:
        b = self.board
        terminal = b.is_terminal()
        black, white = b.scores()
        moves, grids = self._replay()
        return {
            "grid": self._grid(b),
            "turn": _side(b.player),
            "human_color": _side(self.human_color),
            "your_turn": (not terminal) and b.player == self.human_color,
            "legal_actions": [] if terminal else _legal_actions(b),
            "must_pass": (not terminal) and not b.legal_moves(),
            "game_over": terminal,
            "winner": (_side(b.winner()) if b.winner() != 0 else "draw") if terminal else None,
            "score": {"black": black, "white": white},
            "history": [_san(a) for a in self.history],
            "history_actions": list(self.history),
            "moves": moves,
            "positions": grids,              # board after each ply; index 0 = start
            "last_bot_moves": [_san(a) for a in self.last_bot_moves],
            "ply": len(self.history),
            "level": self.level,
        }


def _legal_actions(board: Board) -> List[int]:
    moves = board.legal_moves()
    if moves:
        return [r * 8 + c for r, c in moves]
    return [] if board.is_terminal() else [PASS_ACTION]

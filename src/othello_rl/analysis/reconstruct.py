"""Reconstruct every position of a VALID game with the project engine."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional

from othello_rl.environment.board import PASS_ACTION, Board, action_to_rc
from othello_rl.ingest.records import GameRecord


@dataclass
class Position:
    board: Board
    player: int                 # BLACK / WHITE
    move_number: int            # 0-based ply index within the game
    played_move: int            # action index actually played
    legal_moves: List[int]      # action indices
    game_id: str
    game_result: Optional[str]  # "black" / "white" / "draw" / None
    total_plies: int
    remaining_plies: int        # plies left after this one (>= 1)

    @property
    def side(self) -> str:
        return "black" if self.player == 1 else "white"


def positions(record: GameRecord) -> Iterator[Position]:
    placements = [int(m) for m in (record.canonical_moves or record.moves)
                  if int(m) != PASS_ACTION]
    n = len(placements)
    result = (record.result or {}).get("winner")
    state = Board.initial()
    for i, m in enumerate(placements):
        legal = [r * 8 + c for r, c in state.legal_moves()]
        yield Position(board=state, player=state.player, move_number=i,
                       played_move=m, legal_moves=legal, game_id=record.game_id,
                       game_result=result, total_plies=n, remaining_plies=n - i)
        state = state.apply(action_to_rc(m))

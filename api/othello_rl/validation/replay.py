"""Replay an ingested :class:`GameRecord` move-by-move and classify it.

    VALID              every stored placement was legal when played and the game
                       reaches a terminal position
    INVALID            an illegal placement, or placements recorded after the
                       game already ended
    INCOMPLETE         all stored placements are legal but the game stops before
                       a terminal position
    UNSUPPORTED_FORMAT the record could not be interpreted at all (no moves)

Note on passes: the project's :class:`~othello_rl.environment.board.Board` **auto-
skips** a player with no legal move (``Board.apply`` advances to whoever can
move). Historical formats such as WThor also omit forced passes. So a pass never
appears as an explicit action here — an explicit ``PASS_ACTION`` token in a source
is treated as annotation and dropped. ``canonical_moves`` is the pure placement
sequence that reconstructs the game via ``Board.apply``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from othello_rl.environment.board import Board, PASS_ACTION, action_to_rc
from othello_rl.ingest.records import GameRecord


class Status(str, Enum):
    VALID = "VALID"
    INVALID = "INVALID"
    INCOMPLETE = "INCOMPLETE"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"


@dataclass
class ValidationResult:
    status: Status
    canonical_moves: List[int] = field(default_factory=list)  # placements only
    final_black: Optional[int] = None
    final_white: Optional[int] = None
    plies_replayed: int = 0
    passes_skipped: int = 0
    reason: str = ""
    recorded_winner: Optional[str] = None
    replayed_winner: Optional[str] = None

    @property
    def winner_matches(self) -> Optional[bool]:
        if self.recorded_winner is None or self.replayed_winner is None:
            return None
        return self.recorded_winner == self.replayed_winner


def _winner_str(board: Board) -> str:
    w = board.winner()
    return "black" if w == 1 else "white" if w == -1 else "draw"


def validate(record: GameRecord) -> ValidationResult:
    moves = list(record.moves)
    if not moves or all(int(m) == PASS_ACTION for m in moves):
        return ValidationResult(Status.UNSUPPORTED_FORMAT, reason="no placements")

    state = Board.initial()
    canonical: List[int] = []
    passes = 0
    recorded_winner = (record.result or {}).get("winner")

    for idx, raw in enumerate(moves):
        m = int(raw)
        if m == PASS_ACTION:
            passes += 1                       # engine auto-handles passes
            continue
        if state.is_terminal():
            return ValidationResult(
                Status.INVALID, canonical, plies_replayed=len(canonical),
                passes_skipped=passes,
                reason=f"placement #{idx} recorded after the game ended",
                recorded_winner=recorded_winner)
        rc = action_to_rc(m)
        if rc is None or rc not in state.legal_moves():
            return ValidationResult(
                Status.INVALID, canonical, plies_replayed=len(canonical),
                passes_skipped=passes,
                reason=f"illegal move {m} at ply {len(canonical)}",
                recorded_winner=recorded_winner)
        canonical.append(m)
        state = state.apply(rc)

    if not state.is_terminal():
        return ValidationResult(
            Status.INCOMPLETE, canonical, plies_replayed=len(canonical),
            passes_skipped=passes,
            reason="stored moves stop before a terminal position",
            recorded_winner=recorded_winner)

    b, w = state.scores()
    replayed = _winner_str(state)
    return ValidationResult(
        Status.VALID, canonical, final_black=b, final_white=w,
        plies_replayed=len(canonical), passes_skipped=passes,
        recorded_winner=recorded_winner, replayed_winner=replayed,
        reason="" if recorded_winner in (None, replayed)
        else f"recorded winner {recorded_winner!r} != replayed {replayed!r}")

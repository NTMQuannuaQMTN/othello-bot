"""One analysed game -> a stream of :class:`TrainingExample` (position, played
move, game outcome, label at the chosen horizon). Positions are reconstructed
with the project engine from the ordered ``played_move`` list."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

from othello_rl.environment.board import Board, action_to_rc
from othello_rl.environment.environment import encode_observation

LABEL_CODES = {"BEST": 0, "GOOD": 1, "ACCEPTABLE": 2, "MISTAKE": 3, "BLUNDER": 4}
LABEL_NAMES = {v: k for k, v in LABEL_CODES.items()}
DATA_KIND_CODES = {"historical": 0, "self_play": 1, "engine_generated": 2}


@dataclass
class TrainingExample:
    obs: np.ndarray            # (3, 8, 8) float32, canonical for the side to move
    policy_target: int         # action index actually played (0..63)
    value_target: float        # +1 / 0 / -1 from the mover's perspective
    weight: float
    label: str                 # move-quality label at the chosen horizon
    game_id: str
    move_number: int
    data_kind: str
    source: str


def _z(result_winner: Optional[str], mover_side: str) -> float:
    if result_winner is None:
        return 0.0
    if result_winner == "draw":
        return 0.0
    return 1.0 if result_winner == mover_side else -1.0


def examples_from_analyzed_game(game: dict, horizon: int,
                                weight_fn) -> Iterator[TrainingExample]:
    """``game`` is a line from ``analyzed_games/<source>.jsonl``. ``weight_fn`` is
    ``label -> Optional[float]`` (None => drop this example)."""
    winner = (game.get("result") or {}).get("winner")
    source = game.get("source", "unknown")
    data_kind = game.get("data_kind", "historical")
    hk = str(horizon)

    state = Board.initial()
    for mv in game["moves"]:
        played = int(mv["played_move"])
        j = mv["by_horizon"].get(hk)
        label = j["label"] if j else "ACCEPTABLE"
        w = weight_fn(label)
        if w is not None:
            yield TrainingExample(
                obs=encode_observation(state).astype(np.float32),
                policy_target=played,
                value_target=_z(winner, mv["player"]),
                weight=float(w), label=label, game_id=str(game["game_id"]),
                move_number=int(mv["move_number"]), data_kind=data_kind, source=source)
        state = state.apply(action_to_rc(played))

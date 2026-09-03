"""Counterfactual comparison: played move vs every legal alternative, judged by a
shallow search. ``regret = best_alternative_value − played_move_value``.

**Not an oracle.** The leaf is a static heuristic and the horizon is 3–5 plies;
Othello is positional and a short search can be wrong. Labels are a QC signal for
weighting training data, nothing more.
"""
from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from othello_rl.agents.heuristic_agent import DEFAULT_WEIGHTS
from othello_rl.environment.board import action_to_rc, square_name

from .reconstruct import Position
from .search import move_value

LABELS = ("BEST", "GOOD", "ACCEPTABLE", "MISTAKE", "BLUNDER")

#: default thresholds on ``tanh(regret / evaluation_scale)`` (0..1). Overridable
#: from ``configs/analysis.yaml``; documented in ``docs/historical-training.md``.
DEFAULT_MOVE_QUALITY = {"best": 0.02, "good": 0.06, "acceptable": 0.15, "mistake": 0.35}
DEFAULT_SCALE = 6.0


def classify(regret_norm: float, mq: Optional[dict] = None) -> str:
    mq = mq or DEFAULT_MOVE_QUALITY
    if regret_norm <= mq["best"]:
        return "BEST"
    if regret_norm <= mq["good"]:
        return "GOOD"
    if regret_norm <= mq["acceptable"]:
        return "ACCEPTABLE"
    if regret_norm <= mq["mistake"]:
        return "MISTAKE"
    return "BLUNDER"


@dataclass
class MoveJudgement:
    game_id: str
    move_number: int
    player: str
    played_move: int
    best_move: int
    played_move_value: float
    best_move_value: float
    regret: float               # heuristic-eval units (best − played)
    regret_norm: float          # tanh(max(0, regret) / evaluation_scale), 0..1
    label: str
    lookahead_plies: int
    n_alternatives: int         # legal moves actually searched
    n_legal: int
    evaluation_scale: float
    sampled: bool = False       # were alternatives sub-sampled?
    move_values: Dict[str, float] = field(default_factory=dict)  # san -> value

    def as_dict(self) -> dict:
        return asdict(self)


def _candidate_moves(pos: Position, max_alternatives: int) -> tuple:
    legal = list(pos.legal_moves)
    if not max_alternatives or len(legal) <= max_alternatives:
        return legal, False
    others = [m for m in legal if m != pos.played_move]
    rng = random.Random(hash((pos.game_id, pos.move_number, max_alternatives)))
    keep = rng.sample(others, max_alternatives - 1)
    return [pos.played_move] + keep, True


def judge_position(pos: Position, horizon: int, *, weights: Optional[dict] = None,
                   move_quality: Optional[dict] = None, scale: float = DEFAULT_SCALE,
                   alpha_beta: bool = True, transposition: bool = True,
                   max_alternatives: int = 0) -> MoveJudgement:
    weights = weights or DEFAULT_WEIGHTS
    cands, sampled = _candidate_moves(pos, max_alternatives)
    tt: dict = {} if transposition else None

    values: Dict[int, float] = {}
    for m in cands:
        values[m] = move_value(pos.board, action_to_rc(m), horizon,
                               weights, alpha_beta=alpha_beta, tt=tt)

    best_move = max(values, key=values.get)
    played_v = values[pos.played_move]
    best_v = values[best_move]
    regret = best_v - played_v
    regret_norm = math.tanh(max(0.0, regret) / scale)
    return MoveJudgement(
        game_id=pos.game_id, move_number=pos.move_number, player=pos.side,
        played_move=pos.played_move, best_move=best_move,
        played_move_value=round(played_v, 4), best_move_value=round(best_v, 4),
        regret=round(regret, 4), regret_norm=round(regret_norm, 4),
        label=classify(regret_norm, move_quality),
        lookahead_plies=horizon, n_alternatives=len(cands),
        n_legal=len(pos.legal_moves), evaluation_scale=scale, sampled=sampled,
        move_values={square_name(action_to_rc(m)): round(v, 3)
                     for m, v in sorted(values.items(), key=lambda kv: -kv[1])[:5]},
    )

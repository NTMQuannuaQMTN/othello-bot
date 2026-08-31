"""Short-horizon counterfactual move analysis for historical games.

For every reconstructed position, a shallow negamax (alpha-beta) search estimates
the value of the played move and of every legal alternative ``lookahead_plies``
deep; ``regret = best_alternative_value − played_move_value`` maps to a quality
label. **This is a heuristic labelling / quality-control mechanism, not an Othello
oracle** — the leaf evaluation is a static heuristic and the horizon is tiny.
"""
from .reconstruct import Position, positions
from .search import shallow_value, move_value
from .counterfactual import MoveJudgement, judge_position, classify, LABELS
from .pipeline import analyze_game, analyze_file, benchmark

__all__ = ["Position", "positions", "shallow_value", "move_value",
           "MoveJudgement", "judge_position", "classify", "LABELS",
           "analyze_game", "analyze_file", "benchmark"]

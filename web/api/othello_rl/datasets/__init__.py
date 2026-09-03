"""Turn analysed historical games into versioned, game-level-split training sets.

Three configurable strategies (``configs/dataset.yaml``):

* ``all``       — every valid move, weight 1 (baseline).
* ``filtered``  — keep only BEST / GOOD / ACCEPTABLE moves.
* ``weighted``  — per-label weights (BLUNDER -> 0).

Splitting is at the **game** level (``datasets/split.py``) so no position from a
game ever appears in more than one split.
"""
from .split import SPLITS, assign_split, split_counts
from .examples import TrainingExample, examples_from_analyzed_game
from .build import DatasetConfig, build_dataset

__all__ = ["SPLITS", "assign_split", "split_counts", "TrainingExample",
           "examples_from_analyzed_game", "DatasetConfig", "build_dataset"]

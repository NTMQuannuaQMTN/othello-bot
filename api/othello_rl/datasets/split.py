"""Deterministic game-level train / val / test assignment.

A game's split is a pure function of ``(game_id, seed)`` — stable as the corpus
grows and identical across machines. Every position of a game inherits the
game's split, so there is no leakage of "later moves of a training game" into
val/test.
"""
from __future__ import annotations

import hashlib
from typing import Dict, Iterable

SPLITS = ("train", "val", "test")
_RESOLUTION = 1_000_000


def _fraction(game_id: str, seed: int) -> float:
    h = hashlib.sha1(f"{seed}:{game_id}".encode("utf-8")).digest()
    return int.from_bytes(h[:6], "big") / (1 << 48)


def assign_split(game_id: str, ratios: Dict[str, float], seed: int = 0) -> str:
    train = float(ratios.get("train", 0.8))
    val = float(ratios.get("val", 0.1))
    x = _fraction(str(game_id), seed)
    if x < train:
        return "train"
    if x < train + val:
        return "val"
    return "test"


def split_counts(game_ids: Iterable[str], ratios: Dict[str, float], seed: int = 0
                 ) -> Dict[str, int]:
    out = {s: 0 for s in SPLITS}
    for g in game_ids:
        out[assign_split(g, ratios, seed)] += 1
    return out

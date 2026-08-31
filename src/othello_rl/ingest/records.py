"""The internal game representation shared by every ingestion source.

One ``GameRecord`` per historical game. ``moves`` are flat action indices
(``row * 8 + col``, ``64`` = pass) from the standard initial position, exactly as
the source stores them — most sources (e.g. WThor) omit forced passes; those are
inserted later by :mod:`othello_rl.validation.replay`, which also fills
``canonical_moves``. See ``docs/game-data-format.md``.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

PASS_ACTION = 64

#: where a record came from, kept on every training example downstream
DATA_KINDS = ("historical", "self_play", "engine_generated")


@dataclass
class GameRecord:
    source: str                      # "wthor", "webapp", "transcript", ...
    source_format: str               # "wtb", "jsonl", "transcript", "json"
    moves: List[int]                 # placements as stored by the source (0..63, 64=pass)
    game_id: str = ""                # stable id; defaults to a content hash
    data_kind: str = "historical"
    metadata: Dict[str, Any] = field(default_factory=dict)   # date, tournament, players, ratings…
    result: Optional[Dict[str, Any]] = None   # {"black_discs", "white_discs", "winner"} | None
    provenance: Dict[str, Any] = field(default_factory=dict)  # file, offset, pass_convention…
    ingested_at: str = ""
    canonical_moves: Optional[List[int]] = None   # filled by validation (passes inserted)

    def __post_init__(self) -> None:
        self.moves = [int(m) for m in self.moves]
        if self.canonical_moves is not None:
            self.canonical_moves = [int(m) for m in self.canonical_moves]
        if self.data_kind not in DATA_KINDS:
            raise ValueError(f"data_kind must be one of {DATA_KINDS}, got {self.data_kind!r}")
        if not self.ingested_at:
            self.ingested_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        if not self.game_id:
            self.game_id = f"{self.source}:{self.content_hash()}"

    # -- identity ---------------------------------------------------------
    def move_signature(self) -> str:
        """Move sequence as a compact string — the dedup key (passes ignored)."""
        return ",".join(str(m) for m in self.moves if m != PASS_ACTION)

    def content_hash(self) -> str:
        h = hashlib.sha1(self.move_signature().encode("ascii"))
        return h.hexdigest()[:16]

    # -- serialisation --------------------------------------------------
    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_json(cls, line: str) -> "GameRecord":
        d = json.loads(line)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

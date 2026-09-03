"""Ingest a generic JSON file: a list of game objects, or ``{"games": [...]}``.

Each game object: ``{"moves": [...], "id"?, "metadata"?, "result"?}`` where a move
is an int action index or a square name (``"f5"`` / ``"pass"``).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from othello_rl.environment.board import parse_square

from ..records import GameRecord, PASS_ACTION
from .base import GameSource, UnsupportedFormat


def _to_action(m) -> int:
    if isinstance(m, (int, float)):
        return int(m)
    s = str(m).strip().lower()
    if s in ("pass", "--"):
        return PASS_ACTION
    r, c = parse_square(s)
    return r * 8 + c


class GenericJSONSource(GameSource):
    format_name = "json"
    suffixes = (".json",)

    def parse_file(self, path: Path) -> Iterator[GameRecord]:
        try:
            doc = json.loads(Path(path).read_text())
        except json.JSONDecodeError as e:
            raise UnsupportedFormat(f"{path}: invalid JSON ({e})")
        games = doc.get("games", doc) if isinstance(doc, dict) else doc
        if not isinstance(games, list):
            raise UnsupportedFormat(f"{path}: expected a list of games")
        for i, g in enumerate(games):
            if not isinstance(g, dict) or not g.get("moves"):
                continue
            try:
                moves = [_to_action(m) for m in g["moves"]]
            except ValueError:
                continue
            yield GameRecord(
                source=str(g.get("source", "generic")), source_format="json",
                moves=moves, game_id=str(g.get("id") or f"generic:{Path(path).stem}:{i:04d}"),
                data_kind=str(g.get("data_kind", "historical")),
                metadata=dict(g.get("metadata", {})),
                result=g.get("result"),
                provenance={"file": Path(path).name, "index": i,
                            "pass_convention": "explicit"},
            )

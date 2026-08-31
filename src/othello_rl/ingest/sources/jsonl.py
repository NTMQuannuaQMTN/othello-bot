"""Ingest our own ``data/games.jsonl`` (web-app games) and compatible JSONL.

Each line: ``{"moves": [...], "human_color": "...", "winner": "...",
"score": {"black": n, "white": n}, "bot_version": n, ...}``. These already carry
forced passes, so ``moves`` is used verbatim.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from ..records import GameRecord
from .base import GameSource, UnsupportedFormat


class JsonlSource(GameSource):
    format_name = "jsonl"
    suffixes = (".jsonl",)

    def parse_file(self, path: Path) -> Iterator[GameRecord]:
        text = Path(path).read_text()
        any_line = False
        for ln_no, line in enumerate(text.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                if not any_line:
                    raise UnsupportedFormat(f"{path}: not JSONL")
                continue
            any_line = True
            moves = d.get("moves") or d.get("actions")
            if not moves:
                continue
            score = d.get("score") or {}
            result = None
            if "black" in score and "white" in score:
                b, w = int(score["black"]), int(score["white"])
                result = {"black_discs": b, "white_discs": w,
                          "winner": d.get("winner")
                          or ("black" if b > w else "white" if w > b else "draw")}
            md = {k: v for k, v in d.items()
                  if k not in ("moves", "actions", "score")}
            yield GameRecord(
                source="webapp", source_format="jsonl", moves=list(moves),
                game_id=f"webapp:{Path(path).stem}:{ln_no:06d}",
                data_kind=str(d.get("data_kind", "self_play")),
                metadata=md, result=result,
                provenance={"file": Path(path).name, "line": ln_no,
                            "pass_convention": "explicit"},
            )

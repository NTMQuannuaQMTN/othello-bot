"""Ingest plain Othello transcripts: one game per line (or one game per file),
moves as ``f5 d6 c3 …`` / ``f5,d6`` / run-together ``f5d6c3…``. ``pass`` allowed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from othello_rl.environment.board import parse_square
from othello_rl.webapp.moves import _split_transcript  # reuse the tokenizer

from ..records import GameRecord, PASS_ACTION
from .base import GameSource, UnsupportedFormat


class TranscriptSource(GameSource):
    format_name = "transcript"
    suffixes = (".txt", ".transcript", ".gam")

    def parse_file(self, path: Path) -> Iterator[GameRecord]:
        text = Path(path).read_text()
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()
                 and not ln.lstrip().startswith("#")]
        if not lines:
            return
        # one game per line if several lines look like transcripts, else whole file
        games = lines if len(lines) > 1 else [" ".join(lines)]
        parsed_any = False
        for i, g in enumerate(games):
            toks = _split_transcript(g)
            moves = []
            ok = True
            for t in toks:
                if t == "pass":
                    moves.append(PASS_ACTION)
                    continue
                try:
                    r, c = parse_square(t)
                except ValueError:
                    ok = False
                    break
                moves.append(r * 8 + c)
            if not ok or len(moves) < 2:
                continue
            parsed_any = True
            yield GameRecord(
                source="transcript", source_format="transcript", moves=moves,
                game_id=f"transcript:{Path(path).stem}:{i:04d}",
                provenance={"file": Path(path).name, "line": i,
                            "pass_convention": "explicit"},
            )
        if not parsed_any:
            raise UnsupportedFormat(f"{path}: no parseable transcript lines")

"""Drive a source parser -> dedup -> JSONL of ``GameRecord``. Parsing only;
legality validation is :mod:`othello_rl.validation`."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .dedup import Deduplicator
from .records import GameRecord
from .sources import UnsupportedFormat, get_source


@dataclass
class IngestStats:
    source: str
    files: int = 0
    parsed: int = 0
    duplicates: int = 0
    written: int = 0
    unsupported_files: List[str] = field(default_factory=list)
    started_at: str = ""
    finished_at: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def ingest(source_name: str, paths, out_path, *, limit: Optional[int] = None,
           dedup: bool = True, dedup_seed: Optional[Deduplicator] = None) -> IngestStats:
    src = get_source(source_name)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dd = dedup_seed or Deduplicator()
    stats = IngestStats(source=source_name, started_at=time.strftime("%Y-%m-%dT%H:%M:%S"))

    if isinstance(paths, (str, Path)):
        paths = [paths]
    paths = list(paths)
    stats.files = len(paths)

    with out_path.open("w") as fh:
        try:
            for rec in src.parse(list(paths)):
                stats.parsed += 1
                if dedup and not dd.is_new(rec):
                    stats.duplicates += 1
                    continue
                fh.write(rec.to_json() + "\n")
                stats.written += 1
                if limit and stats.written >= limit:
                    break
        except UnsupportedFormat as e:
            stats.unsupported_files.append(str(e))

    stats.finished_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    (out_path.with_suffix(".ingest.json")).write_text(
        json.dumps(stats.as_dict(), indent=2) + "\n")
    return stats


def read_records(path) -> "list[GameRecord]":
    return [GameRecord.from_json(ln) for ln in Path(path).read_text().splitlines() if ln.strip()]

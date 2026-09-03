"""Cross-source de-duplication by move sequence (forced passes ignored).

The same game frequently appears in more than one database; two records with an
identical placement sequence are the same game.
"""
from __future__ import annotations

from typing import Iterable, Iterator

from .records import GameRecord


class Deduplicator:
    def __init__(self) -> None:
        self._seen: set = set()
        self.kept = 0
        self.duplicates = 0

    def is_new(self, record: GameRecord) -> bool:
        sig = record.move_signature()
        if sig in self._seen:
            self.duplicates += 1
            return False
        self._seen.add(sig)
        self.kept += 1
        return True

    def filter(self, records: Iterable[GameRecord]) -> Iterator[GameRecord]:
        for r in records:
            if self.is_new(r):
                yield r

    def stats(self) -> dict:
        return {"kept": self.kept, "duplicates": self.duplicates,
                "unique": len(self._seen)}

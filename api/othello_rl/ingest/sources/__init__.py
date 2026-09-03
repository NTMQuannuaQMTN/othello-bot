"""Registry of ingestion sources. Each maps raw files -> ``GameRecord`` stream."""
from __future__ import annotations

from typing import Dict, Type

from .base import GameSource, UnsupportedFormat
from .generic import GenericJSONSource
from .jsonl import JsonlSource
from .transcript import TranscriptSource
from .wthor import WThorSource

SOURCES: Dict[str, Type[GameSource]] = {
    "wthor": WThorSource,
    "jsonl": JsonlSource,        # our own data/games.jsonl format
    "webapp": JsonlSource,
    "transcript": TranscriptSource,
    "generic": GenericJSONSource,
}


def get_source(name: str) -> GameSource:
    try:
        return SOURCES[name]()
    except KeyError:
        raise UnsupportedFormat(f"unknown source {name!r}; known: {sorted(SOURCES)}")


__all__ = ["SOURCES", "get_source", "GameSource", "UnsupportedFormat"]

"""Historical Othello game ingestion.

Source-pluggable: each parser in :mod:`othello_rl.ingest.sources` turns a raw file
into a stream of :class:`~othello_rl.ingest.records.GameRecord` — a single,
documented internal representation (see ``docs/game-data-format.md``). Ingestion
only *parses*; legality/replay validation lives in
:mod:`othello_rl.validation`.
"""
from .records import GameRecord
from .dedup import Deduplicator
from .pipeline import IngestStats, ingest, read_records
from .sources import SOURCES, get_source

__all__ = ["GameRecord", "Deduplicator", "SOURCES", "get_source",
           "ingest", "IngestStats", "read_records"]

"""Base class for ingestion sources."""
from __future__ import annotations

import abc
from pathlib import Path
from typing import Iterator, List, Union

from ..records import GameRecord


class UnsupportedFormat(Exception):
    """Raised when a file cannot be parsed by the chosen source."""


class GameSource(abc.ABC):
    #: short format tag stored on every record
    format_name: str = "unknown"
    #: filename suffixes this source handles (lowercase, with dot)
    suffixes: tuple = ()

    @abc.abstractmethod
    def parse_file(self, path: Path) -> Iterator[GameRecord]:
        """Yield one :class:`GameRecord` per game in ``path``.

        Skip individual unparseable games (do not abort the file); raise
        :class:`UnsupportedFormat` only if the whole file is the wrong format.
        """

    def parse(self, paths: Union[Path, str, List[Union[Path, str]]]) -> Iterator[GameRecord]:
        """Parse a file, a directory (all matching suffixes), or a list."""
        if isinstance(paths, (str, Path)):
            paths = [paths]
        for p in paths:
            p = Path(p)
            if p.is_dir():
                files = sorted(f for f in p.rglob("*")
                               if f.is_file() and (not self.suffixes
                                                   or f.suffix.lower() in self.suffixes))
            else:
                files = [p]
            for f in files:
                yield from self.parse_file(f)

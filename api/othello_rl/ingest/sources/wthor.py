"""Parser for the WThor database (``.wtb``) published by the Fédération Française
d'Othello — http://www.ffothello.org/informatique/la-base-wthor/ .

Freely downloadable static files; no authentication, rate limiting or anti-bot
protection is involved. Place the ``.wtb`` files (and, optionally, the companion
``WTHOR.JOU`` player-name and ``WTHOR.TRN`` tournament-name tables) under
``data/raw/wthor/`` yourself — this project does not download them.

File layout (little-endian)::

    header  16 bytes
      0..3    creation century / year / month / day
      4..7    N1  number of games            (uint32)
      8..9    N2  number of records          (uint16)
      10..11  game year                      (uint16)
      12      board size (8x8 files: 0 or 8)  13 game kind  14 depth  15 reserved

    then N1 records of 68 bytes each:
      0..1  tournament label number  (uint16)
      2..3  black player number      (uint16)
      4..5  white player number      (uint16)
      6     real score  = black disc count at game end (0..64)
      7     theoretical score (perfect play)
      8..67 60 move bytes, each = 10*row + col with row,col in 1..8; 0 => no move

Forced passes are NOT stored. The project engine auto-skips a player with no
legal move (``Board.apply``), so replay just works without them.
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Iterator, List, Optional

from ..records import GameRecord
from .base import GameSource, UnsupportedFormat

_HEADER = 16
_RECORD = 68
_MOVES_OFF = 8
_N_MOVES = 60


def _move_byte_to_action(b: int) -> Optional[int]:
    if b == 0:
        return None
    row, col = divmod(b, 10)
    if not (1 <= row <= 8 and 1 <= col <= 8):
        raise ValueError(f"bad WThor move byte {b}")
    return (row - 1) * 8 + (col - 1)


class WThorSource(GameSource):
    format_name = "wtb"
    suffixes = (".wtb",)

    def __init__(self, names: "Optional[NameTables]" = None):
        self.names = names

    def parse_file(self, path: Path) -> Iterator[GameRecord]:
        raw = Path(path).read_bytes()
        if len(raw) < _HEADER:
            raise UnsupportedFormat(f"{path}: shorter than a WThor header")
        n_games = struct.unpack_from("<I", raw, 4)[0]
        game_year = struct.unpack_from("<H", raw, 10)[0]
        body = raw[_HEADER:]
        if len(body) < _RECORD or len(body) % _RECORD != 0:
            raise UnsupportedFormat(
                f"{path}: body is {len(body)} bytes, not a multiple of the "
                f"{_RECORD}-byte 8x8 WThor record")
        avail = len(body) // _RECORD
        if n_games == 0 or n_games > avail:
            n_games = avail  # tolerate a wrong count in the header

        names = self.names or _load_name_tables(Path(path).parent)
        for i in range(n_games):
            off = i * _RECORD
            rec = body[off:off + _RECORD]
            if len(rec) < _RECORD:
                break
            try:
                yield self._record(rec, path, i, game_year, names)
            except ValueError:
                continue  # skip a single corrupt game, keep the file going

    def _record(self, rec: bytes, path: Path, idx: int, year: int,
                names: "Optional[NameTables]") -> GameRecord:
        trn, black_no, white_no = struct.unpack_from("<HHH", rec, 0)
        real_score = rec[6]
        theoretical = rec[7]
        moves: List[int] = []
        for k in range(_N_MOVES):
            a = _move_byte_to_action(rec[_MOVES_OFF + k])
            if a is None:
                break
            moves.append(a)
        if not moves:
            raise ValueError("empty game")
        black_discs = int(real_score)
        white_discs = 64 - black_discs if 0 <= black_discs <= 64 else None
        winner = None
        if white_discs is not None:
            winner = ("black" if black_discs > white_discs
                      else "white" if black_discs < white_discs else "draw")
        md = {"year": int(year), "tournament_no": int(trn),
              "black_player_no": int(black_no), "white_player_no": int(white_no),
              "theoretical_black_score": int(theoretical)}
        if names:
            md["tournament"] = names.tournament(trn)
            md["black_player"] = names.player(black_no)
            md["white_player"] = names.player(white_no)
        return GameRecord(
            source="wthor", source_format="wtb", moves=moves,
            game_id=f"wthor:{Path(path).stem}:{idx:06d}",
            metadata=md,
            result=None if white_discs is None else
            {"black_discs": black_discs, "white_discs": white_discs, "winner": winner},
            provenance={"file": Path(path).name, "record_index": idx,
                        "pass_convention": "implicit"},
        )


# --------------------------------------------------------------------------- #
# Optional WTHOR.JOU / WTHOR.TRN name tables (fixed-width text, latin-1)
# --------------------------------------------------------------------------- #
class NameTables:
    def __init__(self, players: List[str], tournaments: List[str]):
        self._players = players
        self._tournaments = tournaments

    def player(self, n: int) -> str:
        return self._players[n] if 0 <= n < len(self._players) else f"#{n}"

    def tournament(self, n: int) -> str:
        return self._tournaments[n] if 0 <= n < len(self._tournaments) else f"#{n}"


def _read_name_table(path: Path, width: int) -> List[str]:
    raw = path.read_bytes()[_HEADER:]
    return [raw[i:i + width].split(b"\x00")[0].decode("latin-1", "replace").strip()
            for i in range(0, len(raw) - width + 1, width)]


def _load_name_tables(folder: Path) -> Optional[NameTables]:
    jou = next((p for p in folder.glob("*") if p.suffix.lower() == ".jou"), None)
    trn = next((p for p in folder.glob("*") if p.suffix.lower() == ".trn"), None)
    if not jou and not trn:
        return None
    players = _read_name_table(jou, 20) if jou and jou.is_file() else []
    tournaments = _read_name_table(trn, 26) if trn and trn.is_file() else []
    return NameTables(players, tournaments)

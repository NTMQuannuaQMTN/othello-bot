"""A minimal GTP (Go Text Protocol) client for **Egaroucid for Console**.

Egaroucid for Console (https://www.egaroucid.nyanyan.dev/en/console) speaks GTP
when started with ``-gtp``.  The subset it implements is documented in its source
(``src/console/gtp_command*.hpp``); the commands this client uses:

===================  ==========================================================
``clear_board``      reset to the Othello start position, Black to move
``play <col> <mv>``  make a move for ``<col>`` (``black``/``white``); ``<mv>`` is
                     a square like ``D3`` or ``pass``
``genmove <col>``    Egaroucid picks and plays ``<col>``'s move; replies with the
                     square (e.g. ``F5``) or ``PASS``
``final_score``      ``B<n>`` / ``W<n>`` / ``0`` for the current position
``version`` etc.     identification
===================  ==========================================================

Coordinate format: files ``A``-``H`` = columns 0-7, ranks ``1``-``8`` = rows 0-7
— identical to this project's ``(row, col)`` once you subtract 1 from the rank.
So GTP ``D3`` == project ``(row=2, col=3)`` == project square name ``"d3"``.

This client is deliberately tiny: our own engine (``othello_rl.environment``)
stays the source of truth for the board, legal moves and termination; Egaroucid
is consulted only for *its* moves.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Tuple

Move = Optional[Tuple[int, int]]

#: repo root — ``src/othello_rl/eval_external/egaroucid.py`` -> parents[3]
_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXE_NAME = "Egaroucid_for_Console.out"

#: where the console build usually lands after ``clang++ ... -o bin/Egaroucid_for_Console.out``
_KNOWN_LOCATIONS = (
    str(_REPO_ROOT / "Egaroucid-console_v7.8.1" / "bin" / _EXE_NAME),
    "~/Downloads/Egaroucid-console_v7.8.1/bin/Egaroucid_for_Console.out",
    "~/Downloads/Egaroucid-console/bin/Egaroucid_for_Console.out",
    "/Applications/Egaroucid_for_Console.app/Contents/MacOS/Egaroucid_for_Console",
)

#: directories scanned with ``<dir>/Egaroucid-console*/bin/<exe>`` as a fallback
_SCAN_DIRS = (_REPO_ROOT, Path("~/Downloads").expanduser())


class EgaroucidError(RuntimeError):
    """Egaroucid returned a GTP error (``?  ...``) or the process died."""


def find_egaroucid(explicit: Optional[str] = None) -> Path:
    """Locate the Egaroucid console executable.

    Order: ``explicit`` arg / ``$EGAROUCID_EXE`` -> ``$PATH`` -> known locations
    (repo root ``Egaroucid-console_v7.8.1/bin/``, then ``~/Downloads``) -> a
    shallow ``Egaroucid-console*/bin/`` scan of the repo root and ``~/Downloads``.
    A directory argument is accepted and searched for ``bin/<exe>``.
    """
    explicit = explicit or os.environ.get("EGAROUCID_EXE")
    if explicit:
        p = Path(explicit).expanduser()
        if p.is_dir():
            for hit in (p / "bin" / _EXE_NAME, p / _EXE_NAME):
                if hit.is_file():
                    return hit
        elif p.is_file():
            return p
        raise EgaroucidError(f"Egaroucid executable not found: {p}")

    on_path = shutil.which(_EXE_NAME) or shutil.which("Egaroucid_for_Console")
    if on_path:
        return Path(on_path)

    for cand in _KNOWN_LOCATIONS:
        p = Path(cand).expanduser()
        if p.is_file():
            return p

    for d in _SCAN_DIRS:
        if d.is_dir():
            hits = sorted(d.glob(f"Egaroucid-console*/bin/{_EXE_NAME}"))
            if hits:
                return hits[-1]

    raise EgaroucidError(
        "Could not find the Egaroucid console executable. Put an "
        "'Egaroucid-console_v7.8.1/' folder in the repo root (or ~/Downloads), or "
        "build it:\n"
        "  cd Egaroucid-console_v7.8.1 && xattr -cr .\n"
        "  clang++ -O2 ./src/Egaroucid_for_Console.cpp -o ./bin/Egaroucid_for_Console.out "
        "-mtune=native -pthread -std=c++20 -DHAS_NO_AVX2 -DHAS_ARM_PROCESSOR\n"
        "then pass --egaroucid <path>, set $EGAROUCID_EXE, or put it on your PATH."
    )


def coord_to_gtp(move: Move) -> str:
    """``(row, col)`` -> ``"D3"``; ``None`` -> ``"pass"``."""
    if move is None:
        return "pass"
    row, col = move
    if not (0 <= row < 8 and 0 <= col < 8):
        raise ValueError(f"move off board: {move!r}")
    return f"{chr(ord('A') + col)}{row + 1}"


def gtp_to_coord(token: str) -> Move:
    """``"D3"`` -> ``(row, col)``; ``"pass"`` / ``"PASS"`` -> ``None``."""
    t = token.strip()
    if t.lower() == "pass":
        return None
    if len(t) != 2:
        raise ValueError(f"bad GTP coordinate: {token!r}")
    col = ord(t[0].upper()) - ord("A")
    row = int(t[1]) - 1
    if not (0 <= row < 8 and 0 <= col < 8):
        raise ValueError(f"GTP coordinate out of range: {token!r}")
    return row, col


class EgaroucidEngine:
    """A running Egaroucid-for-Console process driven over GTP on stdin/stdout.

    Use as a context manager::

        with EgaroucidEngine(level=10) as eng:
            eng.clear_board()
            sq = eng.genmove("black")          # -> "D3"
            eng.play("white", (2, 3))          # tell it our reply
    """

    def __init__(self, exe: Optional[str] = None, *, level: int = 10,
                 threads: int = 1, nobook: bool = False,
                 extra_args: Tuple[str, ...] = (), move_timeout: float = 120.0):
        self.exe = find_egaroucid(exe)
        # Egaroucid resolves resources/ (eval, book, hash) relative to the
        # executable's directory, so run it from there.
        self.workdir = self.exe.parent
        self.level = int(level)
        self.threads = int(threads)
        self.move_timeout = float(move_timeout)
        args: List[str] = [str(self.exe), "-gtp", "-level", str(self.level),
                           "-t", str(self.threads), "-hash", "0"]
        if nobook:
            args.append("-nobook")
        args.extend(extra_args)
        self._args = args
        self.proc = subprocess.Popen(
            args, cwd=str(self.workdir),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        # sanity: the engine answers a trivial command
        self.name = self.send("name") or "Egaroucid"
        self.version = self.send("version") or "unknown"
        self.protocol_version = self.send("protocol_version") or "2.0"

    # -- lifecycle ------------------------------------------------------
    def __enter__(self) -> "EgaroucidEngine":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self.proc.stdin.write("quit\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
        for stream in (self.proc.stdin, self.proc.stdout):
            try:
                stream.close()
            except Exception:
                pass

    # -- raw GTP ------------------------------------------------------
    def send(self, command: str) -> str:
        """Send one GTP command, return its result payload (no ``=`` prefix).

        Raises :class:`EgaroucidError` on a ``?`` error line or if the process
        exits.  A response ends at the first blank line (GTP terminates every
        reply with a blank line).
        """
        if self.proc.poll() is not None:
            raise EgaroucidError(f"Egaroucid exited (code {self.proc.returncode}) before '{command}'")
        self.proc.stdin.write(command + "\n")
        self.proc.stdin.flush()

        deadline = time.monotonic() + self.move_timeout
        lines: List[str] = []
        while True:
            if time.monotonic() > deadline:
                self.proc.kill()
                raise EgaroucidError(f"timed out after {self.move_timeout}s waiting for '{command}'")
            raw = self.proc.stdout.readline()
            if raw == "":
                raise EgaroucidError(f"Egaroucid closed stdout during '{command}'")
            line = raw.rstrip("\n")
            if line == "":
                if lines:
                    break
                continue
            lines.append(line)

        head = lines[0]
        if head.startswith("?"):
            raise EgaroucidError(f"'{command}' -> {' '.join(lines).lstrip('? ')}")
        # strip the leading '=' (and an optional numeric id we never send)
        first = head[1:].lstrip()
        body = "\n".join([first] + lines[1:]).strip()
        return body

    # -- game commands ---------------------------------------------
    def clear_board(self) -> None:
        self.send("clear_board")

    def play(self, color: str, move: Move) -> None:
        self.send(f"play {color} {coord_to_gtp(move)}")

    def genmove(self, color: str) -> Move:
        resp = self.send(f"genmove {color}")
        return gtp_to_coord(resp)

    def final_score(self) -> str:
        # NOTE: Egaroucid's `final_score` reports B/W from the side-to-move's
        # perspective (it does not normalise to Black), so it is unreliable as a
        # black/white verdict.  Prefer `final_result`.
        return self.send("final_score")

    def final_result(self) -> str:
        """``gogui-rules_final_result`` — a Black-normalised verdict string, e.g.
        ``"Black wins by 12 points. Final score is B 38 and W 26"``."""
        return self.send("gogui-rules_final_result")

    def showboard(self) -> str:
        return self.send("showboard")

    # -- info -------------------------------------------------------
    def describe(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "protocol_version": self.protocol_version,
            "executable": str(self.exe),
            "level": self.level,
            "threads": self.threads,
            "argv": self._args,
        }

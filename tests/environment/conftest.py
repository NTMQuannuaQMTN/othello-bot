import numpy as np

from othello_rl.environment.board import BLACK, EMPTY, WHITE

_CHARS = {".": EMPTY, "X": BLACK, "O": WHITE, "_": EMPTY, "*": EMPTY}


def make_board(rows):
    """Build an 8x8 int8 board from 8 strings of 8 chars each.

    ``X`` = black, ``O`` = white, ``.`` / ``_`` = empty. Whitespace in a row is
    ignored so rows may be written spaced out.
    """
    if isinstance(rows, str):
        rows = [r for r in rows.strip().splitlines()]
    arr = np.zeros((8, 8), dtype=np.int8)
    assert len(rows) == 8, f"need 8 rows, got {len(rows)}"
    for r, row in enumerate(rows):
        cells = row.split() if " " in row else list(row.strip())
        assert len(cells) == 8, f"row {r} has {len(cells)} cells: {row!r}"
        for c, ch in enumerate(cells):
            arr[r, c] = _CHARS[ch]
    return arr

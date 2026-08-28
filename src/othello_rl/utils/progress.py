"""A tiny progress-bar abstraction.

`make_progress` returns either a real ``tqdm`` bar or a no-op shim with the same
minimal surface (``update``, ``set_description``, ``set_postfix``, ``write``,
``close``, context-manager). This keeps the training code free of ``tqdm``
import guards and works whether output is a terminal, a redirected log file, or
``tqdm`` is not installed.
"""
from __future__ import annotations

import sys
from typing import Optional


class _NullBar:
    def __init__(self, total: Optional[int] = None):
        self.total = total
        self.n = 0

    def update(self, n: int = 1) -> None:
        self.n += n

    def set_description(self, *_a, **_k) -> None:
        pass

    def set_postfix(self, *_a, **_k) -> None:
        pass

    def write(self, msg: str) -> None:
        print(msg, flush=True)

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def make_progress(total: Optional[int] = None, enabled: "bool | str" = "auto",
                  desc: Optional[str] = None):
    """Create a progress bar.

    ``enabled``: ``True`` force on, ``False`` force off, ``"auto"`` (default) on
    only when stdout is an interactive terminal.
    """
    if enabled == "auto":
        enabled = sys.stdout.isatty()
    if not enabled:
        return _NullBar(total)
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        return _NullBar(total)
    return tqdm(total=total, desc=desc, unit="step", dynamic_ncols=True,
               smoothing=0.05)

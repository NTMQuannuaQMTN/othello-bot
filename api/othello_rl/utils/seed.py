"""Global seeding helpers for reproducibility."""
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def seed_everything(seed: Optional[int]) -> Optional[int]:
    """Seed ``random``, ``numpy`` and (if importable) ``torch``.

    Returns the seed used (``None`` is a no-op that returns ``None``).
    """
    if seed is None:
        return None
    seed = int(seed) % (2 ** 32)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch is optional at import time for pure-engine use
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - no GPU in CI
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass
    return seed


def spawn_seed(rng: random.Random) -> int:
    """Draw a fresh 31-bit seed from ``rng`` (for per-game / per-worker seeding)."""
    return rng.randrange(1, 2 ** 31 - 1)

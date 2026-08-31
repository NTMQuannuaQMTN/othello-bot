"""Experiment run directories + metadata capture for reproducibility."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def git_commit() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                      stderr=subprocess.DEVNULL, text=True)
        return out.strip()
    except Exception:  # pragma: no cover - not a git checkout
        return None


def create_run_dir(base: str | Path, tag: str = "run") -> Path:
    base = Path(base)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = base / f"{stamp}_{tag}"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return run_dir


def _versions() -> Dict[str, str]:
    v = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import torch
        v["torch"] = torch.__version__
    except ImportError:  # pragma: no cover
        pass
    try:
        import numpy
        v["numpy"] = numpy.__version__
    except ImportError:  # pragma: no cover
        pass
    return v


def write_metadata(run_dir: str | Path, config: Dict[str, Any],
                   extra: Optional[Dict[str, Any]] = None) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "versions": _versions(),
        "argv": sys.argv,
        "config": config,
        **(extra or {}),
    }
    path = run_dir / "metadata.json"
    path.write_text(json.dumps(meta, indent=2, default=str))
    return path


#: repo-level append-only experiment log — one JSON line per training / eval /
#: promotion run, so experiment history survives even when run dirs are pruned.
_REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_INDEX = _REPO_ROOT / "experiments" / "index.jsonl"


def _index_path(path=None) -> Path:
    if path is not None:
        return Path(path)
    return Path(os.environ.get("OTHELLO_EXPERIMENT_INDEX", EXPERIMENT_INDEX))


def log_experiment(row: Dict[str, Any], path=None) -> Path:
    """Append one experiment record to ``experiments/index.jsonl`` (committed).
    ``$OTHELLO_EXPERIMENT_INDEX`` overrides the location (used in tests)."""
    path = _index_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"timestamp": datetime.now().isoformat(timespec="seconds"),
             "git_commit": git_commit(), **row}
    with path.open("a") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")
    return path


def read_experiments(path=None) -> list:
    path = _index_path(path)
    if not path.is_file():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]

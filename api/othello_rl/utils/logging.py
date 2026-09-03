"""Lightweight metric logging: append-only JSONL plus in-memory rows, with CSV
export and a plotting helper. No TensorBoard dependency."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class MetricLogger:
    def __init__(self, path: Optional[Union[str, Path]] = None, echo: bool = False):
        self.path = Path(path) if path else None
        self.echo = echo
        self.rows: List[Dict[str, Any]] = []
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("")  # truncate / create

    def log(self, **row: Any) -> Dict[str, Any]:
        self.rows.append(row)
        if self.path:
            with self.path.open("a") as fh:
                fh.write(json.dumps(row, default=_json_default) + "\n")
        if self.echo:
            print(" ".join(f"{k}={_fmt(v)}" for k, v in row.items()))
        return row

    def filter(self, **eq: Any) -> List[Dict[str, Any]]:
        return [r for r in self.rows if all(r.get(k) == v for k, v in eq.items())]

    def series(self, x: str, y: str, **eq: Any):
        rows = [r for r in self.filter(**eq) if x in r and y in r]
        return [r[x] for r in rows], [r[y] for r in rows]

    def to_csv(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        keys: List[str] = []
        for r in self.rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(self.rows)
        return path

    @staticmethod
    def load(path: Union[str, Path]) -> List[Dict[str, Any]]:
        return [json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip()]


def _json_default(o: Any) -> Any:
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    return str(o)


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)

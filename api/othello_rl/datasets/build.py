"""Build a versioned training dataset from analysed games."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .examples import (DATA_KIND_CODES, LABEL_CODES, LABEL_NAMES,
                       TrainingExample, examples_from_analyzed_game)
from .split import SPLITS, assign_split

_STRATEGIES = ("all", "filtered", "weighted")
_DEFAULT_KEEP = ["BEST", "GOOD", "ACCEPTABLE"]
_DEFAULT_WEIGHTS = {"BEST": 1.0, "GOOD": 0.8, "ACCEPTABLE": 0.5,
                    "MISTAKE": 0.15, "BLUNDER": 0.0}


@dataclass
class DatasetConfig:
    strategy: str = "weighted"
    horizon: int = 5
    label_weights: Dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_WEIGHTS))
    filtered_keep: List[str] = field(default_factory=lambda: list(_DEFAULT_KEEP))
    drop_zero_weight: bool = True
    split: Dict[str, float] = field(
        default_factory=lambda: {"train": 0.8, "val": 0.1, "test": 0.1})
    split_seed: int = 20260901
    sources: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.strategy not in _STRATEGIES:
            raise ValueError(f"strategy must be one of {_STRATEGIES}")

    def weight_fn(self):
        if self.strategy == "all":
            return lambda label: 1.0
        if self.strategy == "filtered":
            keep = set(self.filtered_keep)
            return lambda label: 1.0 if label in keep else None
        w = {**_DEFAULT_WEIGHTS, **self.label_weights}
        drop = self.drop_zero_weight
        return lambda label: (None if (drop and w.get(label, 0.0) <= 0.0)
                              else float(w.get(label, 0.0)))

    def config_hash(self) -> str:
        return hashlib.sha1(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:8]


def _iter_analyzed(paths: List[Path]):
    for p in paths:
        for line in Path(p).read_text().splitlines():
            line = line.strip()
            if line:
                yield json.loads(line)


def build_dataset(analyzed_paths, out_root, cfg: DatasetConfig, *,
                  version: Optional[str] = None, limit_games: Optional[int] = None,
                  report_dir: Optional[Path] = None) -> dict:
    analyzed_paths = [Path(p) for p in ([analyzed_paths] if isinstance(analyzed_paths, (str, Path))
                                        else analyzed_paths)]
    version = version or f"{time.strftime('%Y%m%d-%H%M%S')}_{cfg.config_hash()}"
    out_dir = Path(out_root) / version
    out_dir.mkdir(parents=True, exist_ok=True)
    wfn = cfg.weight_fn()

    buckets: Dict[str, List[TrainingExample]] = {s: [] for s in SPLITS}
    game_split: Dict[str, str] = {}
    n_games = 0
    for game in _iter_analyzed(analyzed_paths):
        if limit_games and n_games >= limit_games:
            break
        n_games += 1
        gid = str(game["game_id"])
        sp = assign_split(gid, cfg.split, cfg.split_seed)
        game_split[gid] = sp
        for ex in examples_from_analyzed_game(game, cfg.horizon, wfn):
            buckets[sp].append(ex)

    manifest = {
        "version": version, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": asdict(cfg), "n_games": n_games,
        "inputs": [str(p) for p in analyzed_paths],
        "label_legend": LABEL_NAMES, "data_kind_legend": {v: k for k, v in DATA_KIND_CODES.items()},
        "splits": {},
    }
    split_games = {}
    for sp in SPLITS:
        exs = buckets[sp]
        _write_split(out_dir / f"{sp}.npz", exs)
        split_games[sp] = {e.game_id for e in exs}
        manifest["splits"][sp] = {
            "n_games": len(split_games[sp]),
            "n_positions": len(exs),
            "label_histogram": _hist(e.label for e in exs),
            "data_kind_histogram": _hist(e.data_kind for e in exs),
            "mean_weight": round(float(np.mean([e.weight for e in exs])), 4) if exs else 0.0,
        }
    # no game may appear in more than one split
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            assert not (split_games[a] & split_games[b]), "split leakage!"

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if report_dir is not None:
        Path(report_dir).mkdir(parents=True, exist_ok=True)
        (Path(report_dir) / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _hist(items) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for it in items:
        out[it] = out.get(it, 0) + 1
    return out


def _write_split(path: Path, exs: List[TrainingExample]) -> None:
    n = len(exs)
    obs = np.zeros((n, 3, 8, 8), dtype=np.float32)
    policy = np.zeros(n, dtype=np.int16)
    value = np.zeros(n, dtype=np.float32)
    weight = np.zeros(n, dtype=np.float32)
    label = np.zeros(n, dtype=np.int8)
    data_kind = np.zeros(n, dtype=np.int8)
    move_no = np.zeros(n, dtype=np.int16)
    gids: List[str] = []
    gid_index: Dict[str, int] = {}
    game_idx = np.zeros(n, dtype=np.int32)
    for i, e in enumerate(exs):
        obs[i] = e.obs
        policy[i] = e.policy_target
        value[i] = e.value_target
        weight[i] = e.weight
        label[i] = LABEL_CODES.get(e.label, 2)
        data_kind[i] = DATA_KIND_CODES.get(e.data_kind, 0)
        move_no[i] = e.move_number
        if e.game_id not in gid_index:
            gid_index[e.game_id] = len(gids)
            gids.append(e.game_id)
        game_idx[i] = gid_index[e.game_id]
    np.savez_compressed(path, obs=obs, policy=policy, value=value, weight=weight,
                        label=label, data_kind=data_kind, move_number=move_no,
                        game_idx=game_idx, game_ids=np.array(gids, dtype=object))

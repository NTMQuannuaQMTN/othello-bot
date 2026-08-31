#!/usr/bin/env python3
"""Build a versioned supervised-pretraining dataset from analysed games.

    python3 scripts/build_dataset.py --config configs/dataset.yaml

Reads data/processed/analyzed_games/<source>.jsonl for each source in the config,
writes data/processed/training_data/<version>/{train,val,test}.npz + manifest.json
(also copied to experiments/<ts>_dataset_<version>/). Split is at the game level.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_ROOT = Path(__file__).resolve().parents[1]

from othello_rl.datasets import DatasetConfig, build_dataset  # noqa: E402
from othello_rl.utils.config import load_config  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(_ROOT / "configs" / "dataset.yaml"))
    ap.add_argument("--limit-games", type=int, default=None)
    args = ap.parse_args(argv)

    raw = dict(load_config(args.config))
    cfg = DatasetConfig(**{k: raw[k] for k in raw if k in DatasetConfig.__dataclass_fields__})

    analyzed = _ROOT / "data" / "processed" / "analyzed_games"
    paths = [analyzed / f"{s}.jsonl" for s in (cfg.sources or [])]
    missing = [p for p in paths if not p.is_file()]
    if not paths or missing:
        print(f"ERROR: analyzed games not found: {missing or '(no sources configured)'}\n"
              f"       run scripts/analyze_games.py first", file=sys.stderr)
        return 2

    version = f"{datetime.now():%Y%m%d-%H%M%S}_{cfg.config_hash()}"
    out_root = _ROOT / "data" / "processed" / "training_data"
    report_dir = _ROOT / "experiments" / f"{datetime.now():%Y%m%d-%H%M%S}_dataset_{version}"

    manifest = build_dataset(paths, out_root, cfg, version=version,
                             limit_games=args.limit_games, report_dir=report_dir)
    print(f"version   : {version}")
    print(f"games     : {manifest['n_games']}")
    for sp, d in manifest["splits"].items():
        print(f"  {sp:<5} {d['n_games']:>6} games  {d['n_positions']:>8} positions  "
              f"mean w {d['mean_weight']}  {d['label_histogram']}")
    print(f"\n-> {out_root / version}")
    print(f"-> {report_dir}/manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Short-horizon counterfactual analysis of VALID games -> per-move quality labels.

    python3 scripts/analyze_games.py --source wthor --benchmark --limit 100
    python3 scripts/analyze_games.py --source wthor

Output: data/processed/analyzed_games/<source>.jsonl  (one game per line, each
move judged at every horizon in configs/analysis.yaml). The 3-5 ply search is a
HEURISTIC labeller, not an oracle (see docs/historical-training.md).
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

from othello_rl.analysis.pipeline import analyze_file, benchmark  # noqa: E402
from othello_rl.ingest import read_records  # noqa: E402
from othello_rl.utils.config import load_config  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--config", default=str(_ROOT / "configs" / "analysis.yaml"))
    ap.add_argument("--in", dest="inp", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--benchmark", action="store_true",
                    help="time analysis on a sample and exit (no output written)")
    args = ap.parse_args(argv)

    cfg = dict(load_config(args.config))
    inp = Path(args.inp) if args.inp else (
        _ROOT / "data" / "processed" / "validated_games" / f"{args.source}.jsonl")
    if not inp.is_file():
        print(f"ERROR: {inp} not found — run scripts/validate_games.py first", file=sys.stderr)
        return 2

    if args.benchmark:
        recs = read_records(inp)[:args.limit or 100]
        print(json.dumps(benchmark(recs, cfg), indent=2))
        return 0

    out = _ROOT / "data" / "processed" / "analyzed_games" / f"{args.source}.jsonl"
    report_dir = _ROOT / "experiments" / f"{datetime.now():%Y%m%d-%H%M%S}_analyze_{args.source}"
    bar = {"n": 0}

    def prog(i, total):
        if i - bar["n"] >= max(1, total // 20) or i == total:
            bar["n"] = i
            print(f"  {i}/{total} games", end="\r", flush=True)

    stats = analyze_file(inp, out, cfg, limit=args.limit, progress=prog,
                         report_dir=report_dir)
    d = stats.as_dict()
    print(f"\ngames     : {d['games']}")
    print(f"positions : {d['positions']}  ({d['positions_per_sec']}/s)")
    for h, counts in sorted(d["label_counts"].items()):
        tot = sum(counts.values())
        parts = "  ".join(f"{k} {counts.get(k, 0)} ({counts.get(k, 0) / tot:.0%})"
                          for k in ("BEST", "GOOD", "ACCEPTABLE", "MISTAKE", "BLUNDER"))
        print(f"@{h} plies: {parts}")
    print(f"\n-> {out}")
    print(f"-> {report_dir}/analysis.stats.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

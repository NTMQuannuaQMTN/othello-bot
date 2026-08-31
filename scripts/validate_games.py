#!/usr/bin/env python3
"""Validate ingested games by replaying every move through the project engine.

    python3 scripts/ingest_games.py   --source wthor
    python3 scripts/validate_games.py --source wthor

VALID games -> data/processed/validated_games/<source>.jsonl
everything else -> data/rejected/<source>.jsonl (with a reason)
stats -> alongside the output and copied to experiments/<ts>_validate_<source>/
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_ROOT = Path(__file__).resolve().parents[1]

from othello_rl.validation import validate_file  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--in", dest="inp", default=None,
                    help="ingested JSONL (default: data/processed/validated_games/<source>.raw.jsonl)")
    args = ap.parse_args(argv)

    proc = _ROOT / "data" / "processed" / "validated_games"
    inp = Path(args.inp) if args.inp else proc / f"{args.source}.raw.jsonl"
    if not inp.is_file():
        print(f"ERROR: {inp} not found — run scripts/ingest_games.py first", file=sys.stderr)
        return 2
    valid_out = proc / f"{args.source}.jsonl"
    rejected_out = _ROOT / "data" / "rejected" / f"{args.source}.jsonl"
    report_dir = _ROOT / "experiments" / f"{datetime.now():%Y%m%d-%H%M%S}_validate_{args.source}"

    stats = validate_file(inp, valid_out, rejected_out, source=args.source,
                          report_dir=report_dir)
    d = stats.as_dict()
    print(f"games processed : {d['total']}")
    print(f"  VALID         : {d['valid']}  ({d['valid_fraction']:.1%})")
    print(f"  INVALID       : {d['invalid']}")
    print(f"  INCOMPLETE    : {d['incomplete']}")
    print(f"  UNSUPPORTED   : {d['unsupported']}")
    print(f"  winner mismatch (kept): {d['winner_mismatch']}")
    if d["reasons"]:
        print("reasons:")
        for k, n in sorted(d["reasons"].items(), key=lambda kv: -kv[1]):
            print(f"  {n:>6}  {k}")
    print(f"\nvalid  -> {valid_out}")
    print(f"reject -> {rejected_out}")
    print(f"stats  -> {report_dir}/validation.stats.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

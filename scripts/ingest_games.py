#!/usr/bin/env python3
"""Ingest raw historical game files into a deduplicated ``GameRecord`` JSONL.

Parsing only — legality/replay validation is ``scripts/validate_games.py``.

    # put the WThor .wtb files under data/raw/wthor/ first (see docs/game-data-format.md)
    python3 scripts/ingest_games.py --source wthor
    python3 scripts/ingest_games.py --source transcript --in mygames.txt --limit 500
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_ROOT = Path(__file__).resolve().parents[1]

from othello_rl.ingest import SOURCES, ingest  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, choices=sorted(SOURCES))
    ap.add_argument("--in", dest="inp", default=None,
                    help="file or directory (default: data/raw/<source>/)")
    ap.add_argument("--out", default=None,
                    help="output JSONL (default: data/processed/validated_games/<source>.raw.jsonl)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-dedup", action="store_true")
    args = ap.parse_args(argv)

    inp = Path(args.inp) if args.inp else _ROOT / "data" / "raw" / args.source
    if not inp.exists():
        print(f"ERROR: no input at {inp} — see docs/game-data-format.md for how to "
              f"obtain the data.", file=sys.stderr)
        return 2
    out = Path(args.out) if args.out else (
        _ROOT / "data" / "processed" / "validated_games" / f"{args.source}.raw.jsonl")

    stats = ingest(args.source, inp, out, limit=args.limit, dedup=not args.no_dedup)
    print(f"source        : {stats.source}")
    print(f"parsed        : {stats.parsed}")
    print(f"duplicates    : {stats.duplicates}")
    print(f"written       : {stats.written}  -> {out}")
    if stats.unsupported_files:
        for u in stats.unsupported_files:
            print(f"  unsupported : {u}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

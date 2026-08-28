#!/usr/bin/env python3
"""Track playing strength across the checkpoints of a training run.

Usage:
    python3 scripts/track.py --run experiments/<run_dir> [--games 60] [--baselines random greedy heuristic minimax:2]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from othello_rl.evaluation.tracking import (  # noqa: E402
    discover_checkpoints,
    track_checkpoints,
    write_tracking_plots,
    write_tracking_report,
)
from othello_rl.utils.seed import seed_everything  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="training run directory (uses <run>/checkpoints/*.pt)")
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--rr-games", type=int, default=40, help="games per round-robin pairing")
    ap.add_argument("--baselines", nargs="*", default=["random", "greedy", "heuristic"])
    ap.add_argument("--seed", type=int, default=4242)
    args = ap.parse_args(argv)
    seed_everything(args.seed)

    paths = discover_checkpoints(args.run)
    if not paths:
        print(f"no checkpoints found under {args.run}")
        return 1
    print(f"tracking {len(paths)} checkpoints: {[p.name for p in paths]}")

    baselines = {b: b for b in args.baselines}
    result = track_checkpoints(paths, baselines=baselines, games=args.games,
                               seed=args.seed, round_robin_games=args.rr_games)

    out_dir = Path(args.run) / "tracking"
    write_tracking_report(result, out_dir)
    write_tracking_plots(result, out_dir)
    print(f"\nwrote {out_dir}/tracking.json, tracking.md, *.png")
    for e in result["checkpoints"]:
        vs = " ".join(f"{b}={e['vs'][b]['win_rate']:.2f}" for b in args.baselines)
        print(f"  {e['name']:>24}  steps={e['env_steps']:>7}  {vs}  elo={e['internal_elo']:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

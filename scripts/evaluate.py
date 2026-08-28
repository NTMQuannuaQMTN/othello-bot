#!/usr/bin/env python3
"""Run baseline agent matchups from a YAML config and write a report.

Usage:
    python3 scripts/evaluate.py --config configs/evaluation.yaml [--games N] [--seed S]
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from othello_rl.evaluation.report import generate_report  # noqa: E402
from othello_rl.evaluation.tournament import play_match  # noqa: E402
from othello_rl.utils.config import dump_config, load_config  # noqa: E402
from othello_rl.utils.seed import seed_everything  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/evaluation.yaml")
    ap.add_argument("--games", type=int, default=None, help="override games per matchup")
    ap.add_argument("--seed", type=int, default=None, help="override seed")
    ap.add_argument("--out", default=None, help="override output directory")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    num_games = args.games if args.games is not None else int(cfg.get("num_games", 100))
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 0))
    opening_plies = int(cfg.get("opening_plies", 4))
    base_out = Path(args.out or cfg.get("output_dir", "experiments"))
    seed_everything(seed)

    run_dir = base_out / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_eval"
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_config({"config_file": str(args.config), "num_games": num_games,
                 "seed": seed, "matchups": cfg["matchups"]}, run_dir / "run_config.yaml")

    matches = []
    for a, b in cfg["matchups"]:
        print(f"  {a} vs {b} ... ", end="", flush=True)
        m = play_match(a, b, num_games=num_games, seed=seed, opening_plies=opening_plies)
        matches.append(m)
        print(f"A {m.a_wins}-{m.b_wins}-{m.draws} (win rate {m.a_win_rate:.3f})")

    results = generate_report(matches, out_dir=run_dir, seed=seed)
    print(f"\nWrote {run_dir}/results.json, report.md, matchups.png")
    print(f"Internal Elo: " + ", ".join(f"{k}={v:.0f}" for k, v in
          sorted(results["elo"].items(), key=lambda kv: -kv[1])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

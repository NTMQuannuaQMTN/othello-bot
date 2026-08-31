#!/usr/bin/env python3
"""Evaluate a candidate checkpoint and promote it to production **only** if it
clears the promotion criterion. This is the *only* way the production model and
`checkpoints/registry.json` ever change — training and the web app never touch
them.

    python3 scripts/promote_model.py checkpoints/experiments/v003.pt \
        --name v003_selfplay --parent v001_curriculum_selfplay \
        --method "self-play from v001" --games 200

Promotion criterion (also in docs/training-and-models.md):

    wilson_lb(win_rate vs current best) > --min-vs-best (0.50)   AND
    win_rate vs random >= best_vs_random - --slack               AND
    win_rate vs greedy >= best_vs_greedy - --slack

`--force` promotes regardless (recorded as "forced": true). If the criterion is
not met nothing is written and the exit code is 1.
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

from othello_rl.evaluation.metrics import summarize_match, wilson_interval  # noqa: E402
from othello_rl.evaluation.tournament import play_match  # noqa: E402
from othello_rl.rl.checkpoint import Registry, load_checkpoint, resolve_checkpoint  # noqa: E402
from othello_rl.utils.seed import seed_everything  # noqa: E402

_PANEL = ["random", "greedy", "heuristic", "minimax:2"]


def _winrate(agent, opp, games, seed):
    m = play_match(agent, opp, num_games=games, seed=seed, opening_plies=4)
    s = summarize_match(m)
    return {"win_rate": s.a_win_rate, "ci_low": s.ci_low, "ci_high": s.ci_high,
            "wins": s.a_wins, "losses": s.b_wins, "draws": s.draws}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("candidate", help="path / 'latest' / 'vNNN' of the candidate checkpoint")
    ap.add_argument("--name", default=None, help="version label to record (default: file stem)")
    ap.add_argument("--parent", default=None)
    ap.add_argument("--method", default="unspecified")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260831)
    ap.add_argument("--min-vs-best", type=float, default=0.50,
                    help="Wilson lower bound on win rate vs current best must exceed this")
    ap.add_argument("--slack", type=float, default=0.03,
                    help="allowed regression vs the current best's win rate vs random/greedy")
    ap.add_argument("--force", action="store_true", help="promote even if the criterion fails")
    args = ap.parse_args(argv)
    seed_everything(args.seed)

    reg = Registry.load()
    cand_path = resolve_checkpoint(args.candidate)
    if not cand_path.is_file():
        print(f"ERROR: candidate not found: {cand_path}", file=sys.stderr)
        return 2
    version = args.name or cand_path.stem
    parent = args.parent or reg.model_version

    cand = load_checkpoint(cand_path).build_agent()
    cand.name = "candidate"

    best_path = reg.active_checkpoint_path()
    has_best = best_path.is_file()
    print(f"candidate : {cand_path}  (-> {version})")
    print(f"current   : {reg.model_version}  ({best_path if has_best else 'none'})")
    print(f"eval      : {args.games} games/opponent, seed {args.seed}\n")

    results = {}
    for i, opp in enumerate(_PANEL):
        results[opp] = _winrate(cand, opp, args.games, args.seed + 11 * i)
        print(f"  vs {opp:<12} {results[opp]['win_rate']:.3f}  "
              f"[{results[opp]['ci_low']:.2f}, {results[opp]['ci_high']:.2f}]")

    vs_best = None
    if has_best:
        best_agent = load_checkpoint(best_path).build_agent()
        best_agent.name = reg.model_version
        vs_best = _winrate(cand, best_agent, args.games, args.seed + 999)
        results["vs_best"] = vs_best
        print(f"  vs {'best':<12} {vs_best['win_rate']:.3f}  "
              f"[{vs_best['ci_low']:.2f}, {vs_best['ci_high']:.2f}]")

    # -- criterion ---------------------------------------------------------
    prev_eval = reg.data.get("evaluation", {}) or {}
    reasons = []
    if vs_best is not None:
        lb = vs_best["ci_low"]
        if lb <= args.min_vs_best:
            reasons.append(f"win rate vs best Wilson-LB {lb:.3f} <= {args.min_vs_best}")
    for opp in ("random", "greedy"):
        prev = prev_eval.get(f"win_rate_vs_{opp}")
        if prev is not None and results[opp]["win_rate"] < prev - args.slack:
            reasons.append(f"regressed vs {opp}: {results[opp]['win_rate']:.3f} < "
                           f"{prev:.3f} - {args.slack}")

    passed = not reasons
    print()
    if passed:
        print("PROMOTION CRITERION: PASS")
    else:
        print("PROMOTION CRITERION: FAIL")
        for r in reasons:
            print(f"  - {r}")

    if not passed and not args.force:
        print("\nnothing written — production model unchanged.")
        return 1

    evaluation = {f"win_rate_vs_{k}": v["win_rate"] for k, v in results.items() if k != "vs_best"}
    if vs_best is not None:
        evaluation["win_rate_vs_best"] = vs_best["win_rate"]
    ck = load_checkpoint(cand_path)
    reg.promote(cand_path, version=version, parent=parent, method=args.method,
                evaluation=evaluation, training_games=ck.games_played,
                trained_env_steps=ck.meta.env_steps, seed=args.seed,
                forced=bool(not passed and args.force))

    # keep a curated copy + an audit trail
    models_copy = _ROOT / "models" / f"othello_bot_{version}.pt"
    import shutil
    shutil.copyfile(cand_path, models_copy)
    _append_models_md(version, parent, args.method, evaluation)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    rep_dir = _ROOT / "experiments" / f"{stamp}_promote_{version}"
    rep_dir.mkdir(parents=True, exist_ok=True)
    (rep_dir / "promotion.json").write_text(json.dumps(
        {"version": version, "parent": parent, "method": args.method,
         "results": results, "passed": passed, "forced": bool(args.force and not passed)},
        indent=2) + "\n")

    print(f"\nPROMOTED -> {version}")
    print(f"  checkpoints/production/best.pt + latest.pt + registry.json updated")
    print(f"  {models_copy.relative_to(_ROOT)}")
    print(f"  {rep_dir.relative_to(_ROOT)}/promotion.json")
    return 0


def _append_models_md(version, parent, method, evaluation) -> None:
    md = _ROOT / "models" / "MODELS.md"
    row = (f"| {version} | {datetime.now():%Y-%m-%d} | {parent or '-'} | {method} | "
           f"{evaluation.get('win_rate_vs_random', float('nan')):.2f} | "
           f"{evaluation.get('win_rate_vs_greedy', float('nan')):.2f} | "
           f"{evaluation.get('win_rate_vs_heuristic', float('nan')):.2f} | "
           f"{evaluation.get('win_rate_vs_minimax:2', float('nan')):.2f} | "
           f"{evaluation.get('win_rate_vs_best', float('nan')):.2f} | |\n")
    if md.is_file():
        md.write_text(md.read_text().rstrip("\n") + "\n" + row)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Play the trained Othello RL bot against **Egaroucid for Console** over GTP.

Examples
--------
One debug game (RL as Black), every move printed::

    python3 scripts/play_egaroucid.py

A 10-game mini-match, colours alternating, results saved under results/egaroucid/::

    python3 scripts/play_egaroucid.py --games 10

Pin a checkpoint / engine strength / executable::

    python3 scripts/play_egaroucid.py --checkpoint checkpoints/production/best.pt \
        --games 10 --level 15 --egaroucid ~/Downloads/Egaroucid-console_v7.8.1/bin/Egaroucid_for_Console.out

The RL model is loaded **once** at start-up and kept in memory; it is never
retrained or modified here.  Our own engine referees legality / passing /
termination (PROJECT_SPEC); Egaroucid is asked only for its own moves.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from othello_rl.environment.board import Board  # noqa: E402
from othello_rl.eval_external import EgaroucidEngine, run_match  # noqa: E402
from othello_rl.rl.checkpoint import Registry, resolve_checkpoint  # noqa: E402
from othello_rl.utils.seed import seed_everything  # noqa: E402
from othello_rl.webapp.bot_service import OthelloBot  # noqa: E402


def _resolve_checkpoint(spec):
    """--checkpoint > checkpoints/registry.json active model. Never a random net."""
    reg = Registry.load()
    if spec:
        p = Path(spec)
        if not p.exists():
            p = resolve_checkpoint(spec)
        return p, f"--checkpoint ({spec})", reg
    return reg.active_checkpoint_path(), f"registry ({reg.model_version})", reg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=None,
                    help="checkpoint path/spec (default: active model in checkpoints/registry.json)")
    ap.add_argument("--games", type=int, default=1, help="number of games (colours alternate)")
    ap.add_argument("--level", type=int, default=10, help="Egaroucid level (0-60, default 10)")
    ap.add_argument("--threads", type=int, default=1, help="Egaroucid search threads")
    ap.add_argument("--opening-plies", type=int, default=4,
                    help="random opening plies for game diversity (default 4)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--egaroucid", default=None, help="path to Egaroucid_for_Console.out")
    ap.add_argument("--nobook", action="store_true", help="run Egaroucid without its opening book")
    ap.add_argument("--start-color", choices=("black", "white"), default="black",
                    help="RL bot's colour in game 1 (default black)")
    ap.add_argument("--move-timeout", type=float, default=120.0,
                    help="seconds to wait for one Egaroucid move")
    ap.add_argument("--out-dir", default="results/egaroucid", help="where match JSON is written")
    ap.add_argument("--no-save", action="store_true", help="do not write result files")
    ap.add_argument("--quiet", action="store_true", help="summary only, no per-move log")
    args = ap.parse_args(argv)

    seed_everything(args.seed)

    # -- load the trained model ONCE ------------------------------------
    ckpt_path, origin, _reg = _resolve_checkpoint(args.checkpoint)
    if not Path(ckpt_path).is_file():
        print(f"ERROR: checkpoint not found: {ckpt_path} (from {origin})", file=sys.stderr)
        return 2
    bot = OthelloBot.load(str(ckpt_path))
    info = bot.info()
    # the human-facing model version lives in the registry; the fine-tune
    # counter (bot.version) rides in the checkpoint meta and is 0 for a
    # curriculum/self-play checkpoint that never went through the web app.
    if not args.checkpoint or str(ckpt_path) == str(_reg.active_checkpoint_path()):
        info["version"] = _reg.model_version
    elif bot.version:
        info["version"] = f"v{bot.version} (fine-tuned, parent {info['parent']})"
    else:
        info["version"] = Path(args.checkpoint).stem
    opening = bot.select_action(Board.initial())
    legal0 = {r * 8 + c for r, c in Board.initial().legal_moves()}
    if opening not in legal0:
        print(f"ERROR: loaded model returned an illegal opening move ({opening})", file=sys.stderr)
        return 3
    print(f"Loaded OthelloRL model: {info['version']}  (parent {info['parent']})")
    print(f"Checkpoint: {ckpt_path}  [{origin}]")
    print(f"  net: {info['network']}  ·  {info['params']:,} params  ·  "
          f"trained env-steps {info['train_env_steps']:,}  ·  baseline {info['baseline']}")
    print(f"  opening move OK ({opening} = {_san(opening)})")

    # -- start Egaroucid ----------------------------------------------
    try:
        engine = EgaroucidEngine(args.egaroucid, level=args.level, threads=args.threads,
                                 nobook=args.nobook, move_timeout=args.move_timeout)
    except Exception as exc:
        print(f"ERROR: could not start Egaroucid: {exc}", file=sys.stderr)
        return 4
    desc = engine.describe()
    print(f"\nEgaroucid: {desc['name']} {desc['version']}")
    print(f"  exe: {desc['executable']}")
    print(f"  GTP protocol {desc['protocol_version']}  ·  level {desc['level']}  ·  "
          f"{desc['threads']} thread(s)  ·  book {'off' if args.nobook else 'on'}")

    verbose = not args.quiet
    try:
        # one game is just a 1-game match, so `--games 1` == game 1 of `--games N`
        summary = run_match(bot, engine, games=args.games,
                            opening_plies=args.opening_plies, seed=args.seed,
                            start_color=args.start_color, verbose=verbose)
    finally:
        engine.close()

    _print_summary(summary, args)

    if not args.no_save:
        _save(summary, args, ckpt_path, origin, info, desc)
    return 0


def _san(action: int) -> str:
    from othello_rl.environment.board import PASS_ACTION, action_to_rc, square_name
    return "pass" if action == PASS_ACTION else square_name(action_to_rc(action))


def _print_summary(s, args) -> None:
    print("\n" + "=" * 56)
    print(f"OthelloRL  vs  Egaroucid (level {args.level})")
    print("=" * 56)
    print(f"Games:      {s.games}")
    print(f"Wins:       {s.rl_wins}")
    print(f"Losses:     {s.egaroucid_wins}")
    print(f"Draws:      {s.draws}")
    print(f"Win rate:   {s.win_rate * 100:.1f}%")
    if s.rl_black_games and s.rl_white_games:
        print(f"  as Black: {s.rl_black_wins}/{s.rl_black_games}   "
              f"as White: {s.rl_white_wins}/{s.rl_white_games}")
    print(f"Average RL score:      {s.rl_mean_score:.1f} discs")
    print(f"Average disc diff:     {s.mean_disc_diff:+.1f}  (RL − Egaroucid)")
    print("-" * 56)
    print(f"RL inference — mean {s.inference_ms_mean:.1f} ms · "
          f"median {s.inference_ms_median:.1f} ms · max {s.inference_ms_max:.1f} ms")
    print(f"Total RL thinking time: {s.rl_total_think_s:.2f} s")
    print(f"Total tournament time:  {s.total_wall_s:.1f} s")
    agree = sum(1 for r in s.records if r.egaroucid_agrees)
    print(f"Egaroucid's own final verdict matched our engine on {agree}/{s.games} games")


def _save(summary, args, ckpt_path, origin, model_info, engine_desc) -> None:
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = _ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": {
            "version": model_info["version"],
            "parent": model_info["parent"],
            "checkpoint": str(ckpt_path),
            "checkpoint_origin": origin,
            "network": model_info["network"],
            "params": model_info["params"],
            "trained_env_steps": model_info["train_env_steps"],
        },
        "egaroucid": engine_desc,
        "config": {
            "games": args.games,
            "level": args.level,
            "threads": args.threads,
            "opening_plies": args.opening_plies,
            "seed": args.seed,
            "book": not args.nobook,
            "start_color": args.start_color,
        },
        "summary": {k: v for k, v in summary.to_dict().items() if k != "records"},
        "games": summary.to_dict()["records"],
    }
    match_path = out_dir / f"match_{stamp}.json"
    match_path.write_text(json.dumps(payload, indent=2) + "\n")

    summary_path = out_dir / "summary.json"
    history = []
    if summary_path.is_file():
        try:
            history = json.loads(summary_path.read_text()).get("matches", [])
        except Exception:
            history = []
    history.append({
        "timestamp": payload["timestamp"],
        "file": match_path.name,
        "model_version": model_info["version"],
        "checkpoint": str(ckpt_path),
        "egaroucid_version": engine_desc["version"],
        "level": args.level,
        "games": summary.games,
        "rl_wins": summary.rl_wins,
        "egaroucid_wins": summary.egaroucid_wins,
        "draws": summary.draws,
        "win_rate": summary.win_rate,
        "mean_disc_diff": summary.mean_disc_diff,
        "inference_ms_mean": summary.inference_ms_mean,
        "inference_ms_max": summary.inference_ms_max,
    })
    summary_path.write_text(json.dumps({"matches": history}, indent=2) + "\n")

    print(f"\nsaved: {match_path}")
    print(f"saved: {summary_path}")


if __name__ == "__main__":
    raise SystemExit(main())

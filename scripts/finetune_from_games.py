#!/usr/bin/env python3
"""Offline fine-tuning from recorded human-vs-bot games.

Every game played in the web app is appended to `webapp_state/games.jsonl`
(one JSON object per line: `{moves, human_color, winner, ...}`). This script
replays all of them into the bot's anchored replay buffer, runs one training
pass, and keeps the update only if it doesn't weaken the bot vs a random
opponent (same guardrail as the in-app "Fine-tune").

    python3 scripts/finetune_from_games.py \
        --games webapp_state/games.jsonl \
        --checkpoint models/othello_bot_v1.pt \
        --out models/othello_bot_v2.pt --grad-steps 400
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from othello_rl.utils.seed import seed_everything  # noqa: E402
from othello_rl.webapp.bot_service import FineTuneConfig, OthelloBot  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", default="webapp_state/games.jsonl")
    ap.add_argument("--checkpoint", default="models/othello_bot_v1.pt")
    ap.add_argument("--out", default=None, help="where to write the fine-tuned checkpoint")
    ap.add_argument("--grad-steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--learn", choices=["bot", "both", "black", "white"], default="bot",
                    help="which moves to learn: bot's own (default), both sides, or one colour")
    ap.add_argument("--guardrail-games", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    seed_everything(args.seed)

    games = [json.loads(l) for l in Path(args.games).read_text().splitlines() if l.strip()]
    if not games:
        print(f"no games in {args.games}")
        return 1
    print(f"{len(games)} games from {args.games}")

    ft = FineTuneConfig(guardrail_games=args.guardrail_games)
    if args.grad_steps:
        ft.grad_steps = args.grad_steps
    if args.lr:
        ft.lr = args.lr
    bot = OthelloBot.load(args.checkpoint, ft_config=ft, seed=args.seed)

    rep = bot.finetune_from_games(
        games, learn_color=None if args.learn == "bot" else args.learn,
        progress=lambda i, n: print(f"  built {i}/{n} games", end="\r"))
    print()
    print(f"grad steps      : {rep.grad_steps}")
    print(f"reinforced/pen. : {rep.n_reinforced} / {rep.n_penalised}")
    print(f"TD loss         : {rep.loss_before:.4f} -> {rep.loss_after:.4f}")
    print(f"win rate vs Rand: {rep.winrate_vs_random_before:.3f} -> {rep.winrate_vs_random_after:.3f}")
    print(f"result          : {'ROLLED BACK (made it weaker)' if rep.rolled_back else 'kept — bot v%d' % rep.version}")

    if args.out and not rep.rolled_back:
        bot.agent.save(args.out)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the standard single-checkpoint evaluation protocol (PROJECT_SPEC Phase 8):
one agent vs Random / Greedy / Heuristic / Minimax(1,2,3), with Wilson CIs and an
internal Elo placement.

    python3 scripts/eval_bot.py --checkpoint models/othello_bot_v1.pt
    python3 scripts/eval_bot.py --agent heuristic --games 200
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from othello_rl.evaluation.bot_report import standard_panel_eval, write_bot_report  # noqa: E402
from othello_rl.utils.experiment import write_metadata  # noqa: E402
from othello_rl.utils.seed import seed_everything  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--checkpoint", default=None, help="path to a DQN checkpoint")
    g.add_argument("--agent", default=None, help="a baseline spec (random/greedy/heuristic/minimax:N)")
    ap.add_argument("--games", type=int, default=100, help="games per opponent")
    ap.add_argument("--elo-extra-games", type=int, default=40,
                    help="games for the light random/greedy/heuristic sub-round-robin (0 to skip)")
    ap.add_argument("--seed", type=int, default=20260829)
    ap.add_argument("--out", default="experiments", help="parent dir for the report")
    args = ap.parse_args(argv)
    seed_everything(args.seed)

    if args.checkpoint:
        from othello_rl.rl.agent import DQNAgent
        agent = DQNAgent.from_checkpoint(args.checkpoint)
        agent_name = Path(args.checkpoint).stem
    else:
        agent = args.agent or "greedy"
        agent_name = agent

    print(f"evaluating {agent_name} vs the standard panel "
          f"({args.games} games/opponent) …")
    result = standard_panel_eval(agent, agent_name=agent_name, num_games=args.games,
                                 elo_extra_games=args.elo_extra_games, seed=args.seed)

    from datetime import datetime
    out_dir = Path(args.out) / f"{datetime.now():%Y%m%d-%H%M%S}_eval_{agent_name}"
    write_bot_report(result, out_dir)
    write_metadata(out_dir, {"agent": agent_name, "checkpoint": args.checkpoint,
                             "games": args.games, "seed": args.seed})

    print(f"\nwrote {out_dir}/bot_eval.md, bot_eval.json\n")
    for name in result["panel"]:
        d = result["vs"][name]
        print(f"  vs {name:<12} {d['win_rate']:.3f}  "
              f"[{d['ci_low']:.2f},{d['ci_high']:.2f}]  disc {d['mean_disc_diff']:+.1f}")
    print(f"\n  internal Elo ({result['elo_kind']}, random=1500): "
          f"{result['agent']} = {result['agent_internal_elo']:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

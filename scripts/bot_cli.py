#!/usr/bin/env python3
"""Stable text protocol for an external harness to test the Othello bot.

One request per line on stdin; one response line on stdout.

    genmove <transcript>     -> the bot's move for the position after <transcript>
                                (transcript = moves so far, e.g. "f5d6c3" or
                                "f5 d6 c3"; empty for the opening position).
                                Reply: a move like "d3", or "pass".
    eval <transcript>        -> "<winprob_black> <best_move>"
    name                     -> bot name + version
    quit                     -> exit

Example:
    printf 'genmove f5d6\\nquit\\n' | python3 scripts/bot_cli.py --checkpoint models/othello_bot_v1.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from othello_rl.environment.board import Board, PASS_ACTION, square_name  # noqa: E402
from othello_rl.webapp.bot_service import OthelloBot  # noqa: E402
from othello_rl.webapp.moves import parse_game  # noqa: E402


def _position(transcript: str) -> Board:
    actions = parse_game(transcript.strip()) if transcript.strip() else []
    state = Board.initial()
    for a in actions:
        state = state.apply(None if a == PASS_ACTION else divmod(a, 8))
    return state


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="models/othello_bot_v1.pt")
    args = ap.parse_args(argv)

    bot = OthelloBot.load(args.checkpoint)
    sys.stderr.write(f"# {bot.info()['name']} v{bot.info()['version']} ready\n")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        cmd, _, rest = line.partition(" ")
        cmd = cmd.lower()
        try:
            if cmd == "quit":
                break
            if cmd == "name":
                info = bot.info()
                print(f"{info['name']} v{info['version']} ({info['params']} params)")
            elif cmd == "genmove":
                board = _position(rest)
                a = bot.select_action(board)
                print("pass" if a == PASS_ACTION else square_name(divmod(a, 8)))
            elif cmd == "eval":
                board = _position(rest)
                ev = bot.evaluate_position(board)
                best = ev["moves"][0]["san"] if ev["moves"] else "pass"
                print(f"{ev['winprob_black']:.3f} {best}")
            else:
                print(f"? unknown command: {cmd}")
        except Exception as e:  # noqa: BLE001
            print(f"? error: {e}")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

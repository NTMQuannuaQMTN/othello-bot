#!/usr/bin/env python3
"""Terminal Othello: play against a baseline agent or a trained DQN checkpoint.

Examples:
    python3 scripts/play.py --opponent heuristic --color black
    python3 scripts/play.py --checkpoint experiments/<run>/checkpoints/final.pt --color white
    python3 scripts/play.py --opponent minimax:3            # you are black by default
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from othello_rl.agents import make_agent  # noqa: E402
from othello_rl.environment.board import BLACK, WHITE, Board, square_name, parse_square  # noqa: E402


def load_opponent(args):
    if args.checkpoint:
        from othello_rl.rl.agent import DQNAgent
        return DQNAgent.from_checkpoint(args.checkpoint)
    return make_agent(args.opponent, seed=args.seed)


def prompt_human_move(state: Board) -> object:
    legal = state.legal_moves()
    if not legal:
        input("You have no legal move — press Enter to pass. ")
        return None
    names = ", ".join(square_name(m) for m in legal)
    while True:
        raw = input(f"Your move (legal: {names}) > ").strip().lower()
        if raw in ("q", "quit", "exit"):
            raise SystemExit(0)
        try:
            mv = parse_square(raw)
        except ValueError:
            print("  bad square, e.g. 'd3'")
            continue
        if mv in legal:
            return mv
        print("  not a legal move")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--opponent", default="heuristic",
                    help="baseline spec: random, greedy, heuristic, minimax:<d>")
    ap.add_argument("--checkpoint", default=None, help="path to a DQN checkpoint (overrides --opponent)")
    ap.add_argument("--color", choices=["black", "white", "random"], default="black")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)
    human = {"black": BLACK, "white": WHITE,
             "random": rng.choice([BLACK, WHITE])}[args.color]
    opponent = load_opponent(args)
    print(f"You are {'BLACK (X)' if human == BLACK else 'WHITE (O)'}; "
          f"opponent = {opponent.name}\n")

    state = Board.initial()
    while not state.is_terminal():
        print(state.render())
        b, w = state.scores()
        print(f"  X={b}  O={w}   turn: {'X' if state.player == BLACK else 'O'}")
        if state.player == human:
            move = prompt_human_move(state)
        else:
            move = opponent.select_move(state)
            print(f"  {opponent.name} plays: {square_name(move) if move else 'pass'}")
        state = state.apply(move)
        print()

    print(state.render())
    b, w = state.scores()
    winner = state.winner()
    result = "draw" if winner == 0 else ("you win!" if winner == human else "you lose.")
    print(f"\nFinal: X={b} O={w} — {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

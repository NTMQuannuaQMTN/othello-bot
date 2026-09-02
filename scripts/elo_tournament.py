#!/usr/bin/env python3
"""Round-robin tournament + Elo — our engine vs the baselines vs Egaroucid 0-10.

Plays every pair, fits a single Elo scale over all the games, and writes the
**measured Elo of each Egaroucid level** — the calibrated anchors for the
training ladder (`scripts/train_vs_egaroucid.py --elo-anchors <file>`), instead
of pretending each level is a fixed number of Elo apart.

    python3 scripts/elo_tournament.py                       # ~1-2 h
    python3 scripts/elo_tournament.py --games 8 --eg-levels 0-6

Output:

    results/tournament/tournament_<stamp>.json    full crosstable + ratings
    results/tournament/egaroucid_anchors.json     {level -> Elo}, our_bot Elo
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from itertools import combinations
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from othello_rl.agents import Agent, make_agent  # noqa: E402
from othello_rl.agents.minimax_agent import MinimaxAgent  # noqa: E402
from othello_rl.environment.board import BLACK, WHITE, Board, action_to_rc, rc_to_action  # noqa: E402
from othello_rl.evaluation.elo import DEFAULT_RATING, EloModel  # noqa: E402
from othello_rl.evaluation.tournament import play_game  # noqa: E402
from othello_rl.eval_external import EgaroucidEngine  # noqa: E402
from othello_rl.eval_external.match import play_game as play_vs_egaroucid  # noqa: E402
from othello_rl.rl.checkpoint import Registry, resolve_checkpoint  # noqa: E402
from othello_rl.utils.seed import seed_everything  # noqa: E402
from othello_rl.webapp.bot_service import OthelloBot  # noqa: E402


class EngineAgent(Agent):
    """Our bot: the search engine (`OthelloBot.best_move`)."""

    def __init__(self, bot: OthelloBot, name: str = "engine"):
        self.bot = bot
        self.name = name

    def select_move(self, state: Board):
        d = self.bot.best_move(state)
        return None if d["action"] == 64 else action_to_rc(d["action"])


class _Bot:
    """Adapt any ``Agent`` to the ``select_action(board) -> int`` interface that
    ``eval_external.match.play_game`` expects for the Egaroucid pairings."""

    def __init__(self, agent: Agent):
        self.agent = agent

    def select_action(self, board: Board) -> int:
        mv = self.agent.select_move(board)
        return 64 if mv is None else rc_to_action(*mv)


def play_egaroucid_pair(eng_a, eng_b, *, a_black: bool, opening_plies: int, rng):
    """One game between two Egaroucid engines, our ``Board`` as referee; each
    engine's move is pushed to the other with GTP ``play``.  Returns A's score."""
    eng_a.clear_board()
    eng_b.clear_board()
    board = Board.initial()
    ply = 0
    while not board.is_terminal():
        legal = board.legal_moves()
        color = "black" if board.player == BLACK else "white"
        a_to_move = (board.player == BLACK) == a_black
        if not legal:
            board = board.apply(None)
            ply += 1
            continue
        if ply < opening_plies:
            mv = rng.choice(legal)
            eng_a.play(color, mv)
            eng_b.play(color, mv)
        else:
            mover, other = (eng_a, eng_b) if a_to_move else (eng_b, eng_a)
            mv = mover.genmove(color)
            if mv is None or mv not in legal:
                mv = rng.choice(legal)          # desync guard
            other.play(color, mv)
        board = board.apply(mv)
        ply += 1
    if board.winner() == 0:
        return 0.5
    return 1.0 if (board.winner() == BLACK) == a_black else 0.0


def _parse_range(s: str):
    if "-" in s:
        lo, hi = s.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in s.split(",")]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=None, help="our bot's checkpoint (default: production)")
    ap.add_argument("--engine-budget", type=float, default=0.3)
    ap.add_argument("--engine-endgame", type=int, default=12)
    ap.add_argument("--games", type=int, default=6, help="games per pair (colours alternate)")
    ap.add_argument("--agent-games", type=int, default=20, help="games per pair among the fast agents")
    ap.add_argument("--opening-plies", type=int, default=2)
    ap.add_argument("--minimax-depths", default="1,2,3")
    ap.add_argument("--eg-levels", default="0-10")
    ap.add_argument("--eg-opponents", default="engine,greedy,heuristic,minimax:2,minimax:3",
                    help="which players face Egaroucid (all pairs is expensive)")
    ap.add_argument("--eg-threads", type=int, default=4)
    ap.add_argument("--eg-vs-eg", type=int, default=6,
                    help="games between each ADJACENT pair of Egaroucid levels "
                         "(gives the ladder its rungs); 0 to skip")
    ap.add_argument("--egaroucid", default=None)
    ap.add_argument("--anchor", default="greedy", help="agent pinned to Elo 1500")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/tournament")
    args = ap.parse_args(argv)
    seed_everything(args.seed)

    ckpt = (Path(args.checkpoint) if args.checkpoint and Path(args.checkpoint).exists()
            else resolve_checkpoint(args.checkpoint) if args.checkpoint
            else Registry.load().active_checkpoint_path())
    bot = OthelloBot.load(str(ckpt))
    bot.engine_budget = args.engine_budget
    bot.engine_endgame = args.engine_endgame

    agents = {
        "engine": EngineAgent(bot),
        "random": make_agent("random", seed=args.seed),
        "greedy": make_agent("greedy"),
        "heuristic": make_agent("heuristic"),
    }
    for d in _parse_range(args.minimax_depths):
        agents[f"minimax:{d}"] = MinimaxAgent(depth=d)
    eg_levels = _parse_range(args.eg_levels)
    eg_opps = [o.strip() for o in args.eg_opponents.split(",") if o.strip()]

    out = _ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    records = []            # (a, b, score_a) per game
    cross = {}              # (a, b) -> [w, d, l] from a's view
    t0 = time.monotonic()

    def note(a, b, sa):
        records.append((a, b, sa))
        wdl = cross.setdefault((a, b), [0, 0, 0])
        wdl[0 if sa > 0.6 else 2 if sa < 0.4 else 1] += 1

    # -- 1. agent round-robin (fast; our engine included) ---------------
    names = list(agents)
    for pi, (a, b) in enumerate(combinations(names, 2)):
        n = args.agent_games if "engine" not in (a, b) else args.games
        for g in range(n):
            a_black = g % 2 == 0                      # play_game's 1st arg is Black
            first, second = (agents[a], agents[b]) if a_black else (agents[b], agents[a])
            first.reset(); second.reset()
            gr = play_game(first, second, seed=args.seed * 131 + pi * 997 + g,
                           opening_plies=args.opening_plies)
            if gr.winner == 0:
                sa = 0.5
            else:
                a_won = (gr.winner == BLACK) == a_black
                sa = 1.0 if a_won else 0.0
            note(a, b, sa)
        print(f"[{int(time.monotonic()-t0):>4}s] {a:<12} vs {b:<12}  "
              f"{cross[(a,b)][0]}-{cross[(a,b)][1]}-{cross[(a,b)][2]}")

    # -- 2. Egaroucid levels vs the chosen opponents --------------------
    import random as _random
    want_eg = bool(eg_opps) or (args.eg_vs_eg > 0 and len(eg_levels) > 1)
    engines = {}
    if want_eg:
        for L in eg_levels:
            engines[L] = EgaroucidEngine(args.egaroucid, level=L, threads=args.eg_threads,
                                         move_timeout=max(120.0, 30.0 * L))
    try:
        for L in (eg_levels if want_eg else []):
            eng, egname = engines[L], f"egaroucid:{L}"
            for opp in eg_opps:
                b = _Bot(agents[opp])
                w = d = l = 0
                for g in range(args.games):
                    rec = play_vs_egaroucid(b, eng, rl_is_black=(g % 2 == 0),
                                            game_index=g, opening_plies=args.opening_plies,
                                            verbose=False)
                    sa = {"rl_win": 1.0, "draw": 0.5, "egaroucid_win": 0.0}[rec.result]
                    note(opp, egname, sa)
                    w += sa == 1.0; d += sa == 0.5; l += sa == 0.0
                print(f"[{int(time.monotonic()-t0):>4}s] {opp:<12} vs {egname:<12}  {w}-{d}-{l}")
        # -- 3. adjacent Egaroucid levels, to rank the ladder ----------
        for lo, hi in (zip(eg_levels, eg_levels[1:]) if want_eg and args.eg_vs_eg > 0 else []):
            rng = _random.Random(args.seed + lo)
            w = d = ll = 0
            for g in range(args.eg_vs_eg):
                sa = play_egaroucid_pair(engines[lo], engines[hi], a_black=(g % 2 == 0),
                                         opening_plies=args.opening_plies, rng=rng)
                note(f"egaroucid:{lo}", f"egaroucid:{hi}", sa)
                w += sa == 1.0; d += sa == 0.5; ll += sa == 0.0
            print(f"[{int(time.monotonic()-t0):>4}s] egaroucid:{lo:<3} vs egaroucid:{hi:<3}  {w}-{d}-{ll}")
    finally:
        for eng in engines.values():
            eng.close()

    # -- 3. fit one Elo scale over every game --------------------------
    model = EloModel(k=16.0, anchor=args.anchor).fit(records, passes=60, seed=args.seed)
    board = model.leaderboard()
    our_elo = model.rating("engine")
    eg_elo = {L: round(model.rating(f"egaroucid:{L}"), 1) for L in eg_levels}

    print(f"\n{'=' * 44}\nElo (anchor {args.anchor} = {DEFAULT_RATING:.0f})")
    for name, r in board:
        print(f"  {name:<14} {r:>7.0f}")
    print(f"\nEgaroucid ladder (measured):")
    for L in eg_levels:
        print(f"  level {L:>2}  Elo {eg_elo[L]:>7.1f}")
    print(f"\nour bot (engine, {args.engine_budget}s): Elo {our_elo:.0f}  "
          f"-> sits at Egaroucid level "
          f"{max([L for L in eg_levels if eg_elo[L] <= our_elo] or [eg_levels[0]])}")

    payload = {
        "stamp": stamp, "checkpoint": str(ckpt), "config": vars(args),
        "seconds": round(time.monotonic() - t0, 1),
        "ratings": {n: round(r, 1) for n, r in board},
        "crosstable": {f"{a} vs {b}": {"w": w, "d": d, "l": lo}
                       for (a, b), (w, d, lo) in cross.items()},
    }
    (out / f"tournament_{stamp}.json").write_text(json.dumps(payload, indent=2) + "\n")
    anchors = {"fitted": stamp, "anchor_agent": args.anchor,
               "our_bot_elo": round(our_elo, 1),
               "egaroucid_elo": {str(L): eg_elo[L] for L in eg_levels},
               "ratings": payload["ratings"]}
    (out / "egaroucid_anchors.json").write_text(json.dumps(anchors, indent=2) + "\n")
    print(f"\nwrote {out / f'tournament_{stamp}.json'}\n      {out / 'egaroucid_anchors.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

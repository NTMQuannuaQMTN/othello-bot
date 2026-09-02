#!/usr/bin/env python3
"""Distil the search engine into the policy net over a *gauntlet* of opponents.

`scripts/train_from_engine.py` generates engine-vs-engine games — both sides
play near-optimally, so the net only ever sees "clean" positions and never
learns what to do in the messy ones a weak or wild opponent creates.

This script keeps the same teacher (the bitboard **engine**,
`othello_rl/engine/solver.py`, labels every position with its best move) but
generates the games by having the engine *play a rotating gauntlet*:

    random, greedy, heuristic, minimax:1, minimax:2, minimax:3, egaroucid:<L>

Each position seen in those games — whoever is to move — is labelled with the
engine's move and added to the replay buffer; the Q-head is trained so
``argmax Q`` matches, plus a value head toward the game outcome.

    python3 scripts/train_vs_gauntlet.py --checkpoint <best.pt> --hours 3

Output (git-ignored, storage-light — no game dump):

    checkpoints/experiments/gauntlet_<stamp>/
      latest.pt / best.pt / final.pt
      progress.jsonl        one row per round
      run.json              config + live status + final eval

Never touches `checkpoints/production/` or the registry — the result is a
candidate; evaluate / promote it with `scripts/eval_bot.py` / `promote_model.py`.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

torch.set_num_threads(1)

from othello_rl.agents import make_agent  # noqa: E402
from othello_rl.agents.minimax_agent import MinimaxAgent  # noqa: E402
from othello_rl.engine import bitboard as bb  # noqa: E402
from othello_rl.engine.solver import best_move as engine_best  # noqa: E402
from othello_rl.environment.board import BLACK, Board, action_to_rc  # noqa: E402
from othello_rl.environment.environment import encode_observation, legal_action_mask  # noqa: E402
from othello_rl.evaluation.tournament import play_match  # noqa: E402
from othello_rl.rl.agent import DQNAgent  # noqa: E402
from othello_rl.rl.checkpoint import Registry, load_agent, resolve_checkpoint  # noqa: E402
from othello_rl.rl.network import masked_q  # noqa: E402
from othello_rl.utils.seed import seed_everything  # noqa: E402


def _fmt(s: float) -> str:
    return str(timedelta(seconds=int(s)))


# --------------------------------------------------------------------------- #
# opponents — a uniform ``.move(board) -> (r, c) | None`` / ``.reset()`` face
# --------------------------------------------------------------------------- #
class _AgentOpp:
    def __init__(self, name: str, agent):
        self.name = name
        self.agent = agent

    def reset(self) -> None:
        if hasattr(self.agent, "reset"):
            self.agent.reset()

    def move(self, board: Board):
        return self.agent.select_move(board)


class _EgaroucidOpp:
    """Egaroucid over GTP.  We push *our* moves to it; it tracks its own."""

    def __init__(self, engine, level: int):
        self.engine = engine
        self.level = level
        self.name = f"egaroucid:{level}"

    def reset(self) -> None:
        self.engine.clear_board()

    def push(self, board: Board, move) -> None:
        color = "black" if board.player == BLACK else "white"
        self.engine.play(color, move)

    def move(self, board: Board):
        color = "black" if board.player == BLACK else "white"
        try:
            mv = self.engine.genmove(color)
        except Exception:
            mv = None
        legal = board.legal_moves()
        if mv is None or mv not in legal:
            return random.choice(legal) if legal else None
        return mv


def build_opponents(specs, *, seed, eg_threads, eg_exe):
    opps, engines = [], []
    for spec in specs:
        spec = spec.strip()
        if not spec:
            continue
        if spec.startswith("egaroucid:"):
            from othello_rl.eval_external import EgaroucidEngine
            level = int(spec.split(":", 1)[1])
            eng = EgaroucidEngine(eg_exe, level=level, threads=eg_threads,
                                  move_timeout=max(120.0, 30.0 * level))
            engines.append(eng)
            opps.append(_EgaroucidOpp(eng, level))
        elif spec.startswith("minimax:"):
            opps.append(_AgentOpp(spec, MinimaxAgent(depth=int(spec.split(":", 1)[1]))))
        elif spec == "random":
            opps.append(_AgentOpp(spec, make_agent("random", seed=seed)))
        else:
            opps.append(_AgentOpp(spec, make_agent(spec)))
    return opps, engines


# --------------------------------------------------------------------------- #
# one engine-vs-opponent game; the engine labels every position
# --------------------------------------------------------------------------- #
def generate_game(rng, opp, *, engine_black: bool, budget: float, endgame: int,
                  explore: float):
    board = Board.initial()
    opp.reset()
    tt: dict = {}
    rows = []                                   # (obs, mask, label, mover_is_black)
    while not board.is_terminal():
        legal = board.legal_moves()
        if not legal:
            board = board.apply(None)
            continue
        engine_turn = (board.player == BLACK) == engine_black
        P, O = bb.from_grid(board.array, board.player)
        sq, _val, _meta = engine_best(P, O, time_budget=budget,
                                      endgame_empties=endgame, tt=tt)
        rows.append((encode_observation(board).astype(np.float32),
                     legal_action_mask(board), int(sq), board.player == BLACK))
        if engine_turn:
            if rng.random() < explore:
                r, c = rng.choice(legal)
                mv = (r, c)
            else:
                mv = action_to_rc(int(sq))
            if isinstance(opp, _EgaroucidOpp):
                opp.push(board, mv)
            board = board.apply(mv)
        else:
            mv = opp.move(board)
            if mv is None or mv not in legal:
                mv = rng.choice(legal)
            board = board.apply(mv)
    b, w = board.scores()
    winner = "black" if b > w else "white" if w > b else "draw"
    examples = []
    for obs, mask, label, mover_black in rows:
        z = 0.0 if winner == "draw" else (
            1.0 if (winner == "black") == mover_black else -1.0)
        examples.append((obs, mask, label, float(z)))
    eng_won = None if winner == "draw" else ((winner == "black") == engine_black)
    return examples, eng_won


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=float, default=3.0)
    ap.add_argument("--minutes", type=float, default=None)
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--checkpoint", default=None,
                    help="base net to fine-tune (default: production model)")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--opponents",
                    default="random,greedy,heuristic,minimax:1,minimax:2,minimax:3,egaroucid:1")
    ap.add_argument("--games-per-opp", type=int, default=3,
                    help="engine-vs-opponent games per opponent per round (colours alternate)")
    ap.add_argument("--engine-budget", type=float, default=0.15,
                    help="engine think time per move while generating labels")
    ap.add_argument("--engine-endgame", type=int, default=12)
    ap.add_argument("--explore", type=float, default=0.12,
                    help="prob the engine plays a random legal move (extra position variety)")
    ap.add_argument("--grad-steps", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--value-weight", type=float, default=0.4)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--buffer", type=int, default=150_000)
    ap.add_argument("--eval-games", type=int, default=60)
    ap.add_argument("--eg-eval-every", type=int, default=4,
                    help="run the (slow) Egaroucid eval every N rounds; 0 to skip")
    ap.add_argument("--eg-eval-games", type=int, default=8)
    ap.add_argument("--eg-threads", type=int, default=4)
    ap.add_argument("--egaroucid", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    duration = (args.seconds if args.seconds else
                (args.minutes * 60.0) if args.minutes else args.hours * 3600.0)
    seed_everything(args.seed)
    rng = random.Random(args.seed)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    reg = Registry.load()
    out = Path(args.out) if args.out else _ROOT / "checkpoints" / "experiments" / f"gauntlet_{stamp}"
    out = out if out.is_absolute() else _ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    base_ckpt = (Path(args.checkpoint) if args.checkpoint and Path(args.checkpoint).exists()
                 else resolve_checkpoint(args.checkpoint) if args.checkpoint
                 else reg.active_checkpoint_path())
    if not Path(base_ckpt).is_file():
        print(f"ERROR: base checkpoint not found: {base_ckpt}", file=sys.stderr)
        return 2

    agent = load_agent(base_ckpt) if not args.fresh else DQNAgent(load_agent(base_ckpt).net_config)
    if not isinstance(agent, DQNAgent):
        print("ERROR: base checkpoint is not a DQN net", file=sys.stderr)
        return 2
    net = agent.net
    dev = agent.device
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    has_value = getattr(net, "with_value_head", False)

    base_agent = load_agent(base_ckpt)
    base_version = Path(str(base_ckpt)).stem

    specs = [s.strip() for s in args.opponents.split(",") if s.strip()]
    opps, engines = build_opponents(specs, seed=args.seed, eg_threads=args.eg_threads,
                                    eg_exe=args.egaroucid)
    eg_levels = [o.level for o in opps if isinstance(o, _EgaroucidOpp)]

    print(f"train_vs_gauntlet — {_fmt(duration)}")
    print(f"  base   : {base_version}  ({sum(p.numel() for p in net.parameters()):,} params, "
          f"value_head={has_value})")
    print(f"  engine : {args.engine_budget}s/move, exact from {args.engine_endgame} empties, "
          f"explore {args.explore:.0%}")
    print(f"  opps   : {', '.join(o.name for o in opps)}")
    print(f"  train  : {args.grad_steps} steps/round x batch {args.batch_size}, lr {args.lr}\n")

    prog = (out / "progress.jsonl").open("a", buffering=1)
    run = {"stamp": stamp, "base": str(base_ckpt), "config": vars(args),
           "started": datetime.now().isoformat(timespec="seconds"), "status": "running"}
    (out / "run.json").write_text(json.dumps(run, indent=2) + "\n")

    def _eval(a, n, with_eg=False):
        d = {o: round(play_match(a, o, num_games=n, seed=args.seed + 7,
                                 opening_plies=4).a_win_rate, 3)
             for o in ("random", "greedy", "heuristic", "minimax:1", "minimax:2", "minimax:3")}
        if with_eg and eg_levels:
            from othello_rl.eval_external import EgaroucidEngine
            from othello_rl.eval_external.match import run_match

            class _BotAdapter:
                def __init__(self, ag):
                    self.ag = ag

                def select_action(self, board):
                    mv = self.ag.select_move(board)
                    from othello_rl.environment.board import rc_to_action
                    return 64 if mv is None else rc_to_action(*mv)

                def reset(self):
                    pass

            for L in eg_levels:
                with EgaroucidEngine(args.egaroucid, level=L, threads=args.eg_threads,
                                     move_timeout=max(120.0, 30.0 * L)) as eng:
                    summ = run_match(_BotAdapter(a), eng, games=args.eg_eval_games,
                                     opening_plies=2, verbose=False)
                    d[f"egaroucid:{L}"] = round(
                        (summ.rl_wins + 0.5 * summ.draws) / max(1, summ.games), 3)
        return d

    buf: list = []
    t0 = time.monotonic()
    rnd = 0
    best_score = -1.0
    n_games = 0

    try:
        while time.monotonic() - t0 < duration:
            rnd += 1
            gen_t = time.monotonic()
            wl = {}
            for opp in opps:
                w = 0
                for g in range(args.games_per_opp):
                    exs, eng_won = generate_game(
                        rng, opp, engine_black=(g % 2 == 0),
                        budget=args.engine_budget, endgame=args.engine_endgame,
                        explore=args.explore)
                    buf.extend(exs)
                    n_games += 1
                    w += 1 if eng_won else 0
                wl[opp.name] = w
            if len(buf) > args.buffer:
                buf = buf[-args.buffer:]
            gen_s = time.monotonic() - gen_t

            obs = torch.as_tensor(np.stack([e[0] for e in buf]), device=dev)
            msk = torch.as_tensor(np.stack([e[1] for e in buf]), dtype=torch.bool, device=dev)
            lab = torch.as_tensor([e[2] for e in buf], dtype=torch.long, device=dev)
            val = torch.as_tensor([e[3] for e in buf], dtype=torch.float32, device=dev)
            net.train()
            train_t = time.monotonic()
            ce_sum = acc_sum = 0.0
            for _ in range(args.grad_steps):
                idx = torch.randint(0, len(buf), (args.batch_size,), device=dev)
                if has_value:
                    q, v = net.forward_with_value(obs[idx])
                else:
                    q, v = net(obs[idx]), None
                logits = masked_q(q, msk[idx]) / args.temp
                ce = F.cross_entropy(logits, lab[idx])
                loss = ce
                if v is not None:
                    loss = ce + args.value_weight * F.mse_loss(torch.tanh(v), val[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                opt.step()
                ce_sum += float(ce.item())
                acc_sum += float((logits.argmax(1) == lab[idx]).float().mean().item())
            net.eval()
            train_s = time.monotonic() - train_t

            agent.net = net
            with_eg = args.eg_eval_every and rnd % args.eg_eval_every == 0
            ev = _eval(agent, args.eval_games, with_eg=with_eg)
            score = (ev["random"] + ev["greedy"] + ev["heuristic"] + ev["minimax:1"]) / 4.0
            agent.save(out / "latest.pt", version=f"gauntlet_{stamp}", parent=base_version,
                       method="engine BC vs gauntlet", games=n_games)
            is_best = score > best_score
            if is_best:
                best_score = score
                agent.save(out / "best.pt", version=f"gauntlet_{stamp}", parent=base_version,
                           method="engine BC vs gauntlet", games=n_games)

            elapsed = time.monotonic() - t0
            row = {"round": rnd, "t": round(elapsed, 1), "games": n_games,
                   "examples": len(buf), "ce": round(ce_sum / args.grad_steps, 4),
                   "train_acc": round(acc_sum / args.grad_steps, 3),
                   "engine_wins": wl, "eval": ev, "score": round(score, 3),
                   "best": is_best, "gen_s": round(gen_s, 1), "train_s": round(train_s, 1)}
            prog.write(json.dumps(row) + "\n")
            print(f"[{_fmt(elapsed)}] round {rnd:>3} | {n_games} games / {len(buf)} ex | "
                  f"CE {row['ce']:.3f} acc {row['train_acc']:.2f} | "
                  f"R/G/H {ev['random']:.2f}/{ev['greedy']:.2f}/{ev['heuristic']:.2f} "
                  f"M1/2/3 {ev['minimax:1']:.2f}/{ev['minimax:2']:.2f}/{ev['minimax:3']:.2f}"
                  f"{'  <- BEST' if is_best else ''} | gen {gen_s:.0f}s train {train_s:.0f}s")
            run.update(status="running", rounds=rnd, games=n_games, examples=len(buf),
                       best_score=round(best_score, 3), last_eval=ev, elapsed_s=round(elapsed, 1))
            (out / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    except KeyboardInterrupt:
        pass
    finally:
        for eng in engines:
            try:
                eng.close()
            except Exception:
                pass

    agent.save(out / "final.pt", version=f"gauntlet_{stamp}", parent=base_version)
    print("\nfinal eval (raw policy, no search) …")
    n = max(120, args.eval_games)
    ev = {"base": _eval(base_agent, n, with_eg=bool(args.eg_eval_every)),
          "final": _eval(agent, n, with_eg=bool(args.eg_eval_every))}
    if (out / "best.pt").is_file():
        ev["best"] = _eval(load_agent(out / "best.pt"), n, with_eg=bool(args.eg_eval_every))
    cols = ["random", "greedy", "heuristic", "minimax:1", "minimax:2", "minimax:3"]
    cols += [f"egaroucid:{L}" for L in eg_levels]
    print(f"\n{'model':<7} " + " ".join(f"{c:>10}" for c in cols))
    for k, d in ev.items():
        print(f"{k:<7} " + " ".join(f"{d.get(c, float('nan')):>10.3f}" for c in cols))
    run.update(status="done", ended=datetime.now().isoformat(timespec="seconds"), eval=ev)
    (out / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    prog.close()
    print(f"\ncandidate: {out / 'best.pt'}")
    print(f"  eval:    python3 scripts/eval_bot.py --checkpoint {out / 'best.pt'} --vs-production")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

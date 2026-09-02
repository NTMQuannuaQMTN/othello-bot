#!/usr/bin/env python3
"""Distil the search engine into the policy net (behaviour cloning).

The bitboard **engine** (`othello_rl/engine/solver.py`) is now a genuinely strong
Othello player — it beats a 3-ply minimax 100% and solves the endgame exactly.
This script teaches the DQN to imitate it: generate engine self-play games (the
engine's move at every position is the label), then train the Q-head so
``argmax Q`` matches the engine's move, plus a value head toward the game outcome.

It runs a **generate -> train -> eval** loop for ``--hours``:

    python3 scripts/train_from_engine.py --hours 4

Output (git-ignored):

    checkpoints/experiments/engine_bc_<stamp>/
      games.jsonl        every generated game + per-ply engine labels (resumable)
      latest.pt          the trained candidate (updated each round)
      best.pt            best so far by raw-policy win rate vs Random+Greedy+Heuristic
      progress.jsonl     one row per training round
      run.json           config + live status + final eval

Never touches `checkpoints/production/` or the registry — the result is a
candidate; evaluate / promote it with `scripts/eval_bot.py` / `promote_model.py`.
"""
from __future__ import annotations

import argparse
import json
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

from othello_rl.engine import bitboard as bb  # noqa: E402
from othello_rl.engine.solver import best_move as engine_best  # noqa: E402
from othello_rl.environment.board import Board, action_to_rc  # noqa: E402
from othello_rl.environment.environment import encode_observation, legal_action_mask  # noqa: E402
from othello_rl.evaluation.tournament import play_match  # noqa: E402
from othello_rl.rl.agent import DQNAgent  # noqa: E402
from othello_rl.rl.checkpoint import Registry, load_agent, resolve_checkpoint  # noqa: E402
from othello_rl.rl.network import masked_q  # noqa: E402
from othello_rl.utils.seed import seed_everything  # noqa: E402


def _fmt(s: float) -> str:
    return str(timedelta(seconds=int(s)))


# --------------------------------------------------------------------------- #
# game generation — the engine labels every position it sees
# --------------------------------------------------------------------------- #
def generate_game(rng, *, budget: float, endgame: int, opening_plies: int,
                  explore: float) -> dict:
    """Play one game.  Normally both sides play the engine's move; with prob
    ``explore`` the mover plays a random legal move instead (position variety) —
    but the engine's pick is still recorded as the label for that position."""
    board = Board.initial()
    P, O = bb.from_grid(board.array, board.player)
    plies, labels = [], []
    tt: dict = {}
    ply = 0
    while not board.is_terminal():
        legal = board.legal_moves()
        if not legal:
            board = board.apply(None)
            P, O = bb.from_grid(board.array, board.player)
            continue
        if bb.legal_moves(P, O) == 0:                 # keep bitboard in sync on pass
            P, O = O, P
        sq, _val, _meta = engine_best(P, O, time_budget=budget,
                                      endgame_empties=endgame, tt=tt)
        labels.append(int(sq))                        # the teacher's move
        if ply < opening_plies or rng.random() < explore:
            played = rng.choice([r * 8 + c for (r, c) in legal])
        else:
            played = int(sq)
        plies.append(played)
        board = board.apply(action_to_rc(played))
        P, O = bb.from_grid(board.array, board.player)
        ply += 1
    b, w = board.scores()
    return {"moves": plies, "labels": labels,
            "winner": "black" if b > w else "white" if w > b else "draw",
            "score": [int(b), int(w)]}


def game_to_examples(game: dict):
    """(obs, engine-move label, outcome z from the mover's view) per ply."""
    winner = game["winner"]
    state = Board.initial()
    for played, label in zip(game["moves"], game["labels"]):
        while not state.legal_moves() and not state.is_terminal():
            state = state.apply(None)
        if state.is_terminal():
            break
        z = 0.0 if winner == "draw" else (
            1.0 if (winner == "black") == (state.player == 1) else -1.0)
        yield (encode_observation(state).astype(np.float32),
               legal_action_mask(state), int(label), float(z))
        state = state.apply(action_to_rc(played))


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=float, default=4.0)
    ap.add_argument("--minutes", type=float, default=None)
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--checkpoint", default=None,
                    help="base net to fine-tune (default: production model)")
    ap.add_argument("--fresh", action="store_true",
                    help="reinitialise the net instead of fine-tuning the base")
    ap.add_argument("--engine-budget", type=float, default=0.08,
                    help="engine think time per move while generating labels")
    ap.add_argument("--engine-endgame", type=int, default=10)
    ap.add_argument("--opening-plies", type=int, default=6)
    ap.add_argument("--explore", type=float, default=0.35,
                    help="prob the mover plays a random legal move (position variety)")
    ap.add_argument("--games-per-round", type=int, default=25)
    ap.add_argument("--grad-steps", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--value-weight", type=float, default=0.4)
    ap.add_argument("--temp", type=float, default=1.0, help="softmax temperature on Q for the CE loss")
    ap.add_argument("--buffer", type=int, default=120_000, help="max training examples kept")
    ap.add_argument("--eval-games", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", default=None, help="continue a run dir (reuses games.jsonl + latest.pt)")
    args = ap.parse_args(argv)

    duration = (args.seconds if args.seconds else
                (args.minutes * 60.0) if args.minutes else args.hours * 3600.0)
    seed_everything(args.seed)
    import random
    rng = random.Random(args.seed)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    reg = Registry.load()
    if args.resume:
        out = Path(args.resume)
        out = out if out.is_absolute() else _ROOT / out
        base_ckpt = out / "latest.pt"
    else:
        out = Path(args.out) if args.out else _ROOT / "checkpoints" / "experiments" / f"engine_bc_{stamp}"
        out = out if out.is_absolute() else _ROOT / out
        base_ckpt = (Path(args.checkpoint) if args.checkpoint and Path(args.checkpoint).exists()
                     else resolve_checkpoint(args.checkpoint) if args.checkpoint
                     else reg.active_checkpoint_path())
    out.mkdir(parents=True, exist_ok=True)
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

    base_agent = load_agent(base_ckpt)          # frozen, for the before/after eval
    base_version = reg.model_version if not args.checkpoint else Path(str(base_ckpt)).stem

    games_path = out / "games.jsonl"
    buf: list = []
    n_games = 0
    if games_path.is_file():                    # resume: reload games
        for line in games_path.read_text().splitlines():
            if not line.strip():
                continue
            g = json.loads(line)
            n_games += 1
            for ex in game_to_examples(g):
                buf.append(ex)
        buf = buf[-args.buffer:]
        print(f"  resumed: {n_games} games, {len(buf)} examples")

    print(f"train_from_engine — {_fmt(duration)}")
    print(f"  base   : {base_version}  ({sum(p.numel() for p in net.parameters()):,} params, "
          f"value_head={has_value})")
    print(f"  engine : {args.engine_budget}s/move, exact from {args.engine_endgame} empties, "
          f"explore {args.explore:.0%}")
    print(f"  train  : {args.grad_steps} steps/round x batch {args.batch_size}, lr {args.lr}\n")

    gp = games_path.open("a", buffering=1)
    prog = (out / "progress.jsonl").open("a", buffering=1)
    run = {"stamp": stamp, "base": str(base_ckpt), "config": vars(args),
           "started": datetime.now().isoformat(timespec="seconds"), "status": "running"}
    (out / "run.json").write_text(json.dumps(run, indent=2) + "\n")

    t0 = time.monotonic()
    rnd = 0
    best_score = -1.0

    def _eval(a, n):
        return {o: round(play_match(a, o, num_games=n, seed=args.seed + 7,
                                    opening_plies=4).a_win_rate, 3)
                for o in ("random", "greedy", "heuristic", "minimax:2")}

    try:
        while time.monotonic() - t0 < duration:
            rnd += 1
            gen_t = time.monotonic()
            for _ in range(args.games_per_round):
                g = generate_game(rng, budget=args.engine_budget,
                                  endgame=args.engine_endgame,
                                  opening_plies=args.opening_plies, explore=args.explore)
                gp.write(json.dumps(g) + "\n")
                n_games += 1
                for ex in game_to_examples(g):
                    buf.append(ex)
            if len(buf) > args.buffer:
                buf = buf[-args.buffer:]
            gen_s = time.monotonic() - gen_t

            # -- train on the accumulated engine labels ------------------
            obs = torch.as_tensor(np.stack([e[0] for e in buf]), device=dev)
            msk = torch.as_tensor(np.stack([e[1] for e in buf]), dtype=torch.bool, device=dev)
            lab = torch.as_tensor([e[2] for e in buf], dtype=torch.long, device=dev)
            val = torch.as_tensor([e[3] for e in buf], dtype=torch.float32, device=dev)
            net.train()
            train_t = time.monotonic()
            ce_sum = acc_sum = 0.0
            for step in range(args.grad_steps):
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
            ev = _eval(agent, args.eval_games)
            score = (ev["random"] + ev["greedy"] + ev["heuristic"]) / 3.0
            agent.save(out / "latest.pt", version=f"engine_bc_{stamp}", parent=base_version,
                       method="engine behaviour-cloning", games=n_games)
            is_best = score > best_score
            if is_best:
                best_score = score
                agent.save(out / "best.pt", version=f"engine_bc_{stamp}", parent=base_version,
                           method="engine behaviour-cloning", games=n_games)

            elapsed = time.monotonic() - t0
            row = {"round": rnd, "t": round(elapsed, 1), "games": n_games,
                   "examples": len(buf), "ce": round(ce_sum / args.grad_steps, 4),
                   "train_acc": round(acc_sum / args.grad_steps, 3), "eval": ev,
                   "score": round(score, 3), "best": is_best,
                   "gen_s": round(gen_s, 1), "train_s": round(train_s, 1)}
            prog.write(json.dumps(row) + "\n")
            print(f"[{_fmt(elapsed)}] round {rnd:>3} | {n_games} games / {len(buf)} ex | "
                  f"CE {row['ce']:.3f} acc {row['train_acc']:.2f} | "
                  f"vs R/G/H/M2 {ev['random']:.2f}/{ev['greedy']:.2f}/{ev['heuristic']:.2f}/{ev['minimax:2']:.2f}"
                  f"{'  <- BEST' if is_best else ''} | gen {gen_s:.0f}s train {train_s:.0f}s")
            run.update(status="running", rounds=rnd, games=n_games, examples=len(buf),
                       best_score=round(best_score, 3), last_eval=ev, elapsed_s=round(elapsed, 1))
            (out / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    except KeyboardInterrupt:
        pass

    agent.save(out / "final.pt", version=f"engine_bc_{stamp}", parent=base_version)
    print("\nfinal eval (raw policy, no search) …")
    ev = {"base": _eval(base_agent, max(120, args.eval_games)),
          "final": _eval(agent, max(120, args.eval_games))}
    if (out / "best.pt").is_file():
        ev["best"] = _eval(load_agent(out / "best.pt"), max(120, args.eval_games))
    print(f"\n{'model':<7} {'Random':>8} {'Greedy':>8} {'Heuristic':>10} {'Minimax2':>9}")
    for k, d in ev.items():
        print(f"{k:<7} {d['random']:>8.3f} {d['greedy']:>8.3f} {d['heuristic']:>10.3f} {d['minimax:2']:>9.3f}")
    run.update(status="done", ended=datetime.now().isoformat(timespec="seconds"), eval=ev)
    (out / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    gp.close(); prog.close()
    print(f"\ncandidate: {out / 'best.pt'}")
    print(f"  eval:    python3 scripts/eval_bot.py --checkpoint {out / 'best.pt'} --vs-production")
    print(f"  promote: python3 scripts/promote_model.py {out / 'best.pt'}   # if it earns it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

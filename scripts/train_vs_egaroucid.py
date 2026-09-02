#!/usr/bin/env python3
"""Long, unattended training loop: play Egaroucid, fine-tune on the games, repeat.

    python3 scripts/train_vs_egaroucid.py --hours 8

Each round: play a short match vs **Egaroucid for Console** (GTP), then fine-tune
the model on those games via the project's existing behaviour-cloning path
(`OthelloBot.finetune_from_games` — shaping + an anchored replay buffer + a
**guardrail** that rolls back any update that weakens the bot vs Random).  The
model is loaded once and fine-tuned in place across every round.

**Elo ladder.** The RL bot has an Elo, starting at ``--elo-start`` (500).
The Egaroucid level it faces follows that Elo; after each round the Elo moves by
the standard update (K = ``--elo-k``) on the round's score, drifting to where the
bot is ~even with its level.

* ``--elo-anchors egaroucid_anchors.json`` (from ``scripts/elo_tournament.py``):
  use the **measured** Elo of each Egaroucid level — level = the strongest one the
  bot's Elo is at or above; it faces that level's real rating.  This is the
  calibrated ladder.
* otherwise: ``level = floor(Elo / --elo-band)`` and level N is *assumed* to be
  Elo ``band * (N+1)`` (a rough guess — the levels are actually much closer).

An ``elo_history.png`` graph is written as it goes.

**Move review.** After each round every RL move is graded ``--grade-lookahead``
plies deep: a move that left the bot worse off than the best available is
penalised (``--blunder-penalty``) and the better move reinforced, so the policy
is pushed away from those mistakes. Each round logs how many moves were flagged
and the worst one.

Storage is deliberately tiny — **no per-match result files**.  Only:

    checkpoints/experiments/egaroucid_train_<stamp>/
      latest.pt          most recent kept model               (overwritten)
      best.pt  best.json  best by win% vs Random+Greedy        (overwritten)
      peak_elo.pt         the model at its highest Elo
      snapshots/hNN.pt    one per elapsed hour
      final.pt            the model when the run ends
      progress.jsonl      one compact numeric row per round (elo, level, result)
      elo_history.png     Elo over time
      run.json            config + live status + final eval

Stop early: ``Ctrl-C``, or ``touch <out>/STOP``.  Nothing here ever writes to
`checkpoints/production/` or `checkpoints/registry.json`; the result is a
*candidate* — evaluate / promote it afterwards with
`scripts/eval_bot.py` / `scripts/promote_model.py`.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch  # noqa: E402

# the net is tiny (410k params, 8x8) and every call is a single position — many
# small forwards, where 1 thread beats N (no thread-pool thrash, esp. next to a
# multi-threaded Egaroucid).
torch.set_num_threads(1)

from othello_rl.eval_external import (  # noqa: E402
    EgaroucidEngine, finetune_on_records, records_to_training_games, run_match,
)
from othello_rl.eval_external.match import BestMoveBot  # noqa: E402
from othello_rl.evaluation.tournament import play_match  # noqa: E402
from othello_rl.rl.checkpoint import Registry, resolve_checkpoint  # noqa: E402
from othello_rl.utils.seed import seed_everything  # noqa: E402
from othello_rl.webapp.bot_service import FineTuneConfig, OthelloBot  # noqa: E402


def _parse_duration(hours, minutes, seconds) -> float:
    total = (hours or 0) * 3600.0 + (minutes or 0) * 60.0 + (seconds or 0)
    return total if total > 0 else 8 * 3600.0


def _load_anchors(path):
    """``egaroucid_anchors.json`` from ``scripts/elo_tournament.py`` ->
    ``{level: measured Elo}`` (ints), or ``None``."""
    if not path:
        return None
    d = json.loads(Path(path).read_text())
    return {int(k): float(v) for k, v in d["egaroucid_elo"].items()}


def _level_for_elo(elo: float, band: float, lo: int, hi: int, anchors=None) -> int:
    """Egaroucid level for a bot Elo.  With measured ``anchors`` -> the strongest
    level whose Elo the bot is at or above (clamped).  Otherwise ``floor(elo/band)``."""
    if anchors:
        cand = [L for L in sorted(anchors) if anchors[L] <= elo and lo <= L <= hi]
        return cand[-1] if cand else max(lo, min(hi, min(anchors)))
    return max(lo, min(hi, int(max(0.0, elo) // band)))


def _opp_elo(level: int, band: float, anchors=None) -> float:
    """Elo the RL bot is scored against at ``level`` — the level's **measured**
    Elo (``anchors``), else the top of its band."""
    if anchors and level in anchors:
        return anchors[level]
    if anchors:                                   # clamp to nearest known level
        return anchors[min(anchors, key=lambda L: abs(L - level))]
    return band * (level + 1)


def _elo_update(elo: float, opp_elo: float, score: float, games: int, k: float) -> tuple:
    """One round of Elo. ``score`` = wins + 0.5*draws out of ``games``.
    Returns ``(new_elo, expected_score)`` (expected out of ``games``)."""
    expected = games / (1.0 + 10.0 ** ((opp_elo - elo) / 400.0))
    return max(0.0, elo + k * (score - expected)), expected


def _fmt(sec: float) -> str:
    return str(timedelta(seconds=int(sec)))


def _quick_eval(agent, games: int, seed: int, opps=("random", "greedy", "heuristic")) -> dict:
    out = {}
    for opp in opps:
        m = play_match(agent, opp, num_games=games, seed=seed, opening_plies=4)
        out[opp] = round(m.a_win_rate, 3)
    return out


def _quick_eval2(agent, games: int, seed: int) -> dict:
    return _quick_eval(agent, games, seed, opps=("random", "greedy"))


def _write_elo_plot(out: Path, band: float) -> Optional[Path]:
    """Plot the RL bot's Elo over rounds (and over hours) from progress.jsonl.
    Regenerable at any time — safe to call repeatedly."""
    rows = []
    p = out / "progress.jsonl"
    if not p.is_file():
        return None
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if "elo" in r:
            rows.append(r)
    if len(rows) < 2:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    rounds = [r["round"] for r in rows]
    elo = [r["elo"] for r in rows]
    hours = [r.get("t", 0) / 3600.0 for r in rows]
    level = [r.get("level", 1) for r in rows]
    lvl_max = max(level)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharey=False)
    for ax, x, xl in ((ax1, rounds, "round"), (ax2, hours, "compute hours")):
        for L in range(0, lvl_max + 2):                 # shade + label the level bands
            ax.axhspan(L * band, (L + 1) * band, color="C0" if L % 2 else "C1", alpha=0.06)
            if L * band <= max(elo) + band:
                ax.axhline(L * band, color="0.7", lw=0.6, ls=":")
                ax.text(x[0], L * band + band * 0.04, f"L{L}", fontsize=7, color="0.5")
        ax.plot(x, elo, color="C3", lw=1.4)
        ax.scatter(x, elo, c=level, cmap="viridis", s=10, zorder=3)
        ax.set_xlabel(xl)
        ax.set_ylabel("RL bot Elo (training ladder)")
        ax.set_ylim(0, max(band * 1.2, max(elo) * 1.1))
    ax1.set_title(f"Elo over training — start {rows[0]['elo']:.0f}, "
                  f"now {elo[-1]:.0f}, peak {max(elo):.0f}  "
                  f"(level = floor(Elo / {band:.0f}))")
    fig.tight_layout()
    path = out / "elo_history.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=float, default=None,
                    help="budget of ACTIVE compute time (the clock pauses while the Mac sleeps)")
    ap.add_argument("--minutes", type=float, default=None)
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--wall-hours", type=float, default=None,
                    help="hard wall-clock cap regardless of sleep (default: 3x the compute budget)")
    ap.add_argument("--max-rounds", type=int, default=None, help="also stop after N rounds")
    ap.add_argument("--checkpoint", default=None,
                    help="base checkpoint (default: active model in checkpoints/registry.json)")
    ap.add_argument("--games", type=int, default=8, help="games per round (default 8)")
    ap.add_argument("--best-moves", action=argparse.BooleanOptionalAction, default=True,
                    help="in the match, play the bot's analysed best move (shallow "
                         "search + corner-safety), not the bare policy argmax. Slower "
                         "per round; the fine-tune then clones those stronger moves.")
    ap.add_argument("--level-start", type=int, default=0, help="Egaroucid has a level 0")
    ap.add_argument("--level-end", type=int, default=20, help="level cap (Egaroucid goes to 60)")
    # --- Elo ladder: the RL bot has an Elo; the Egaroucid level it faces is set
    #     by that Elo, and its Elo moves up/down on each round's result -------------
    ap.add_argument("--elo-start", type=float, default=500.0,
                    help="the RL bot's starting Elo (default: lowest anchor if --elo-anchors)")
    ap.add_argument("--elo-anchors", default=None,
                    help="egaroucid_anchors.json from scripts/elo_tournament.py — use the "
                         "MEASURED Elo of each Egaroucid level as the ladder (instead of "
                         "assuming a fixed --elo-band per level)")
    ap.add_argument("--elo-band", type=float, default=1000.0,
                    help="Elo per level: level = floor(elo / band); level N's opponent "
                         "Elo = band * (N+1). So Elo 0-1000 -> level 0, 1000-2000 -> level 1, ...")
    ap.add_argument("--elo-k", type=float, default=24.0,
                    help="Elo K-factor per round (bigger = faster swings)")
    ap.add_argument("--opening-plies", type=int, default=0,
                    help="random opening plies for game variety. 0 = every move is "
                         "the bot's best move from the start (Egaroucid's own jitter "
                         "still gives a few distinct games per round)")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--egaroucid", default=None, help="path to Egaroucid_for_Console.out")
    ap.add_argument("--grad-steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--grade-lookahead", type=int, default=2,
                    help="negamax depth for reviewing each move after the round — how "
                         "far ahead we look to decide a move hurt. Bigger = better "
                         "bad-move detection, slower (1 fast, 3 the app default)")
    ap.add_argument("--blunder-penalty", type=float, default=0.8,
                    help="how hard to push the policy away from a move the review "
                         "flagged as putting the bot in a worse position")
    ap.add_argument("--guardrail-games", type=int, default=40,
                    help="games vs Random for the before/after rollback check "
                         "(lower = faster, noisier)")
    ap.add_argument("--anchor-transitions", type=int, default=3000,
                    help="baseline (bot-vs-Random/Greedy) transitions seeded into the buffer")
    ap.add_argument("--anchor-refill-every", type=int, default=20,
                    help="rounds between topping the anchor (baseline) buffer back up")
    ap.add_argument("--anchor-refill", type=int, default=1500)
    ap.add_argument("--best-eval-every", type=int, default=25,
                    help="rounds between the vs-Random+Greedy check that picks best.pt")
    ap.add_argument("--best-eval-games", type=int, default=100, help="games/opponent for that check")
    ap.add_argument("--eval-games", type=int, default=200, help="games/opponent for the final eval")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="output dir (default checkpoints/experiments/egaroucid_train_<stamp>)")
    ap.add_argument("--resume", default=None, help="resume: continue from <dir>/latest.pt into <dir>")
    args = ap.parse_args(argv)

    duration = _parse_duration(args.hours, args.minutes, args.seconds)
    seed_everything(args.seed)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    anchors = _load_anchors(args.elo_anchors)
    if anchors and args.elo_start == 500.0:          # default -> weakest anchor
        args.elo_start = min(anchors.values())

    def _abs(p):
        p = Path(p)
        return p if p.is_absolute() else (_ROOT / p)

    # -- resolve output dir & base checkpoint (never a random net) --------
    reg = Registry.load()
    if args.resume:
        out = _abs(args.resume)
        base_ckpt = out / "latest.pt"
        base_label = f"resume ({base_ckpt})"
    else:
        out = _abs(args.out) if args.out else \
            _ROOT / "checkpoints" / "experiments" / f"egaroucid_train_{stamp}"
        if args.checkpoint:
            p = Path(args.checkpoint)
            base_ckpt = p if p.exists() else resolve_checkpoint(args.checkpoint)
            base_label = f"--checkpoint ({args.checkpoint})"
        else:
            base_ckpt = reg.active_checkpoint_path()
            base_label = f"registry ({reg.model_version})"
    if not Path(base_ckpt).is_file():
        print(f"ERROR: base checkpoint not found: {base_ckpt}", file=sys.stderr)
        return 2
    (out / "snapshots").mkdir(parents=True, exist_ok=True)
    base_version = Path(str(base_ckpt)).stem if args.checkpoint else reg.model_version

    # -- continue round numbering / counters / Elo when resuming -------
    prev_rounds = prev_kept = prev_rolled = 0
    prev_elo = args.elo_start
    prev_peak_elo = args.elo_start
    prog_path = out / "progress.jsonl"
    if args.resume and prog_path.is_file():
        for line in prog_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            rr = int(r.get("round", 0))
            if rr >= prev_rounds:
                prev_rounds = rr
                prev_elo = float(r.get("elo", args.elo_start))
            prev_peak_elo = max(prev_peak_elo, float(r.get("elo", args.elo_start)))
            if isinstance(r.get("ft"), dict):
                prev_kept += bool(r["ft"].get("kept"))
                prev_rolled += (not r["ft"].get("kept"))
        print(f"  resuming after round {prev_rounds} at Elo {prev_elo:.0f} "
              f"({prev_kept} kept / {prev_rolled} rolled back so far)")
    prev_best = -1.0
    if args.resume and (out / "best.json").is_file():
        try:
            prev_best = float(json.loads((out / "best.json").read_text()).get("score", -1.0))
        except (ValueError, OSError):
            pass

    # -- load the model ONCE --------------------------------------------
    ft = FineTuneConfig(
        grad_steps=args.grad_steps, guardrail_games=args.guardrail_games,
        buffer_capacity=50_000, anchor_transitions=args.anchor_transitions,
    )
    if args.lr:
        ft.lr = args.lr
    bot = OthelloBot.load(str(base_ckpt), ft_config=ft, seed=args.seed)
    n_params = sum(p.numel() for p in bot.agent.net.parameters())

    print(f"train_vs_egaroucid — {_fmt(duration)} compute budget")
    print(f"  base model : {base_version}  [{base_label}]  ({n_params:,} params)")
    print(f"  round      : {args.games} games, "
          f"{args.opening_plies or 'no'} random opening plies; RL plays "
          f"{'its analysed best move (search + corner-safety) every move' if args.best_moves else 'the policy argmax'}")
    print(f"  Elo ladder : start {prev_elo:.0f}  (K={args.elo_k:.0f}); level = "
          f"floor(Elo / {args.elo_band:.0f}), clamped [{args.level_start}, {args.level_end}]; "
          f"level N faces Elo {args.elo_band:.0f}*(N+1). Elo moves on each round's result.")
    print(f"  fine-tune  : {ft.grad_steps} grad steps; after each round every move is "
          f"reviewed {args.grade_lookahead}-ply deep — bad ones penalised "
          f"(x{args.blunder_penalty:g}), the better move reinforced; "
          f"guardrail {ft.guardrail_games} games vs Random")
    print(f"  output     : {out}   (candidate — production untouched)")
    print(f"  stop early : Ctrl-C  or  touch {out / 'STOP'}\n")

    wall_cap = (args.wall_hours * 3600.0) if args.wall_hours else 3.0 * duration

    run_meta = {
        "stamp": stamp, "base_checkpoint": str(base_ckpt), "base_version": base_version,
        "duration_s": duration, "wall_cap_s": wall_cap, "config": vars(args),
        "started": datetime.now().isoformat(timespec="seconds"),
        "resumed_after_round": prev_rounds or None, "status": "running",
    }
    (out / "run.json").write_text(json.dumps(run_meta, indent=2) + "\n")
    progress = prog_path.open("a", buffering=1)

    # -- the loop ----------------------------------------------------
    # `duration` is a budget of ACTIVE compute: time.monotonic() does not advance
    # while the Mac is asleep, so a sleepy laptop just does fewer rounds rather
    # than "finishing" in 8h of mostly-frozen wall time. `wall_cap` stops it for
    # real. Run under `caffeinate` (and keep it on power / lid open) for a true
    # 8-hour run.
    t0 = time.monotonic()
    w0 = time.time()
    deadline = t0 + duration
    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("flag", True))
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("flag", True))

    engine = None
    cur_level = None
    elo = prev_elo
    peak_elo = prev_peak_elo
    level = _level_for_elo(elo, args.elo_band, args.level_start, args.level_end, anchors)
    rnd = prev_rounds
    kept = prev_kept
    rolled = prev_rolled
    errors = 0
    ema_wr = None
    best_wr = prev_best
    hours_saved = set()
    plot_at = 0
    slept_s = 0.0
    match_bot = BestMoveBot(bot) if args.best_moves else bot   # how RL picks moves in the match

    def _open_engine(level: int):
        nonlocal engine, cur_level
        if engine is not None:
            try:
                engine.close()
            except Exception:
                pass
        engine = EgaroucidEngine(args.egaroucid, level=level, threads=args.threads,
                                 move_timeout=max(120.0, 20.0 * level))
        cur_level = level

    def _save(path: Path, **extra):
        bot.agent.save(path, version=f"egaroucid_train_{stamp}", parent=base_version,
                       rounds=rnd, kept_updates=kept, **extra)

    try:
        while not stop["flag"] and time.monotonic() < deadline:
            if (out / "STOP").exists():
                print("STOP file found — finishing up.")
                break
            if args.max_rounds and rnd - prev_rounds >= args.max_rounds:
                break
            if time.time() - w0 >= wall_cap:
                print(f"wall-clock cap ({_fmt(wall_cap)}) reached — stopping.")
                break
            rnd += 1
            level = _level_for_elo(elo, args.elo_band, args.level_start, args.level_end, anchors)
            now_mono = time.monotonic()
            # a big wall gap with almost no monotonic movement == the Mac slept
            gap = (time.time() - w0) - (now_mono - t0)
            if gap - slept_s > 120:
                just_slept = gap - slept_s
                slept_s = gap
                print(f"[{_fmt(now_mono - t0)}] (was asleep ~{_fmt(just_slept)} — "
                      f"compute clock paused; run under `caffeinate` to avoid this)")
            elapsed = now_mono - t0
            if level != cur_level:
                _open_engine(level)
                print(f"[{_fmt(elapsed)}] round {rnd}: Elo {elo:.0f} -> Egaroucid level {level}")

            opp_elo = _opp_elo(level, args.elo_band, anchors)
            row = {"round": rnd, "t": round(elapsed, 1), "level": level,
                   "elo_before": round(elo, 1), "opp_elo": opp_elo}
            try:
                summ = run_match(match_bot, engine, games=args.games,
                                 opening_plies=args.opening_plies,
                                 seed=args.seed + rnd * 7919, verbose=False)
                score = summ.rl_wins + 0.5 * summ.draws
                elo, expected = _elo_update(elo, opp_elo, score, args.games, args.elo_k)
                peak_elo = max(peak_elo, elo)
                row["match"] = {"rl_w": summ.rl_wins, "eg_w": summ.egaroucid_wins,
                                "draw": summ.draws, "disc_diff": round(summ.mean_disc_diff, 1),
                                "rl_score": round(summ.rl_mean_score, 1),
                                "win_rate": round(summ.win_rate, 3)}
                row["elo"] = round(elo, 1)
                row["elo_expected"] = round(expected, 2)      # expected wins out of --games
            except Exception as exc:  # engine hiccup — restart and skip the round
                errors += 1
                row["error"] = f"match: {exc}"
                progress.write(json.dumps(row) + "\n")
                print(f"[{_fmt(elapsed)}] round {rnd}: match error ({exc}); restarting engine")
                try:
                    _open_engine(level)
                except Exception as exc2:
                    print(f"  could not restart Egaroucid: {exc2}")
                    break
                continue

            try:
                rep = finetune_on_records(bot, summ.records, grad_steps=args.grad_steps,
                                          lr=args.lr, guardrail_games=args.guardrail_games,
                                          grade_lookahead=args.grade_lookahead,
                                          blunder_penalty=args.blunder_penalty)
            except Exception as exc:
                errors += 1
                row["error"] = f"finetune: {exc}"
                progress.write(json.dumps(row) + "\n")
                print(f"[{_fmt(elapsed)}] round {rnd}: finetune error ({exc})")
                continue

            # review the round's moves: which ones the look-ahead flagged as bad
            grades = rep.grades or []
            pen = [g for g in grades if g.get("penalised")]
            worst = max(grades, key=lambda g: g.get("drop", 0.0), default=None)
            row["review"] = {
                "graded": len(grades),
                "bad": len(pen),
                "reinforced": sum(bool(g.get("reinforced")) for g in grades),
                "worst": ({"san": worst["played_san"], "instead": worst["best_san"],
                           "ep_lost": round(worst.get("drop", 0.0), 3), "label": worst["label"]}
                          if worst and worst.get("drop", 0) > 0.05 else None),
            }

            was_kept = not rep.rolled_back
            kept += was_kept
            rolled += (not was_kept)
            wr = rep.winrate_vs_random_after
            ema_wr = wr if ema_wr is None else 0.7 * ema_wr + 0.3 * wr
            row["ft"] = {"kept": was_kept, "version": rep.version,
                         "loss": [round(rep.loss_before, 4), round(rep.loss_after, 4)],
                         "wr_rand": [round(rep.winrate_vs_random_before, 3), round(wr, 3)],
                         "wr_rand_ema": round(ema_wr, 3),
                         "reinf": rep.n_reinforced, "pen": rep.n_penalised}

            is_best = False
            try:
                if was_kept:
                    _save(out / "latest.pt")

                # pick best.pt by a real vs-Random+Greedy check, not the noisy guardrail
                due = args.best_eval_every and rnd % args.best_eval_every == 0
                if was_kept and (due or best_wr < 0):
                    bev = _quick_eval2(bot.agent, args.best_eval_games, args.seed + 5)
                    score = 0.5 * bev["random"] + 0.5 * bev["greedy"]
                    row["check"] = bev
                    if score > best_wr + 1e-9:
                        best_wr, is_best = score, True
                        _save(out / "best.pt")
                        (out / "best.json").write_text(json.dumps(
                            {"round": rnd, "elapsed_s": round(elapsed, 1), "score": round(score, 3),
                             "vs": bev, "version": rep.version, "level": level}, indent=2) + "\n")

                if was_kept and elo >= peak_elo - 1e-9:
                    _save(out / "peak_elo.pt", elo=round(elo, 1))

                h = int(elapsed // 3600)
                if h >= 1 and h not in hours_saved:
                    hours_saved.add(h)
                    _save(out / "snapshots" / f"h{h:02d}.pt", elo=round(elo, 1))

                if args.anchor_refill_every and rnd % args.anchor_refill_every == 0:
                    bot._ensure_buffer()
                    bot._fill_anchor(args.anchor_refill)   # top the baseline games back up

                if elapsed - plot_at > 120:            # refresh the Elo graph ~every 2 min
                    plot_at = elapsed
                    _write_elo_plot(out, args.elo_band)
            except Exception as exc:
                errors += 1
                row["bookkeeping_error"] = str(exc)
                print(f"[{_fmt(elapsed)}] round {rnd}: post-finetune error ({exc}); continuing")

            row["best"] = is_best
            progress.write(json.dumps(row) + "\n")
            eta = _fmt(max(0.0, deadline - time.monotonic()))
            if rnd % 5 == 0 or is_best or "check" in row:
                m = row.get("match", {})
                rv = row.get("review", {})
                chk = f" check R/G {row['check']['random']:.2f}/{row['check']['greedy']:.2f}" \
                    if "check" in row else ""
                w = rv.get("worst")
                worst = f" worst {w['san']}→{w['instead']} (-{w['ep_lost']:.2f})" if w else ""
                print(f"[{_fmt(elapsed)}] r{rnd:>4} L{row['level']} | "
                      f"match {m.get('rl_w','?')}-{m.get('eg_w','?')} "
                      f"({m.get('win_rate', 0):.0%}) diff {m.get('disc_diff','?'):>6} | "
                      f"Elo {row['elo_before']:.0f}->{elo:.0f} (exp {row['elo_expected']:.1f}/"
                      f"{args.games}, peak {peak_elo:.0f}) | "
                      f"review {rv.get('bad', 0)} bad/{rv.get('reinforced', 0)} good{worst} | "
                      f"ft {'kept' if was_kept else 'ROLL'}{chk}"
                      f"{'  <- BEST' if is_best else ''} | ETA {eta}")

            run_meta.update(status="running", rounds=rnd, kept=kept,
                            rolled_back=rolled, errors=errors, best_score=round(best_wr, 3),
                            wr_rand_ema=round(ema_wr, 3), current_level=level,
                            elo=round(elo, 1), peak_elo=round(peak_elo, 1),
                            round_win_rate=round(summ.win_rate, 3),
                            compute_s=round(elapsed, 1), wall_s=round(time.time() - w0, 1),
                            slept_s=round(slept_s, 1))
            (out / "run.json").write_text(json.dumps(run_meta, indent=2) + "\n")
    except KeyboardInterrupt:
        stop["flag"] = True
    except Exception as exc:  # never lose the run to an unexpected error — finalise
        print(f"\n[fatal] loop stopped by an unexpected error: {exc!r}\nsaving what we have …")
        run_meta["fatal_error"] = repr(exc)
    finally:
        if engine is not None:
            try:
                engine.close()
            except Exception:
                pass

    # -- finalise --------------------------------------------------
    _save(out / "final.pt")
    # guarantee a best.pt: if nothing was ever kept / beat the base, best == base
    best_pt = out / "best.pt"
    if not best_pt.is_file():
        shutil.copyfile(base_ckpt, best_pt)
        (out / "best.json").write_text(json.dumps(
            {"round": 0, "note": "no round beat the base — best == base", "vs": None}, indent=2) + "\n")
    if not (out / "peak_elo.pt").is_file():          # Elo never rose above its start
        shutil.copyfile(base_ckpt, out / "peak_elo.pt")
    elapsed = time.monotonic() - t0
    new_rounds = rnd - prev_rounds
    print(f"\n{'=' * 60}\ntraining loop ended — {_fmt(elapsed)} compute"
          f"{f' (+{_fmt(slept_s)} asleep)' if slept_s > 60 else ''}, "
          f"{new_rounds} rounds this run / {rnd} total, "
          f"{kept} kept, {rolled} rolled back, {errors} errors")
    print(f"Elo: start {prev_elo:.0f}  ->  end {elo:.0f}   (peak {peak_elo:.0f}, "
          f"highest level {_level_for_elo(peak_elo, args.elo_band, args.level_start, args.level_end)})")
    plot = _write_elo_plot(out, args.elo_band)
    if plot:
        print(f"Elo-over-time graph: {plot}")

    print("\nfinal eval (this takes a couple of minutes) …")
    base_agent = OthelloBot.load(str(base_ckpt)).agent
    ev = {
        "base": _quick_eval(base_agent, args.eval_games, args.seed + 1),
        "final": _quick_eval(bot.agent, args.eval_games, args.seed + 1),
    }
    if best_pt.is_file():
        from othello_rl.rl.checkpoint import load_agent
        ev["best"] = _quick_eval(load_agent(best_pt), args.eval_games, args.seed + 1)

    print(f"\n{'model':<8} {'vs Random':>10} {'vs Greedy':>10} {'vs Heuristic':>13}")
    for name, d in ev.items():
        print(f"{name:<8} {d['random']:>10.3f} {d['greedy']:>10.3f} {d['heuristic']:>13.3f}")

    run_meta.update(status="done", ended=datetime.now().isoformat(timespec="seconds"),
                    rounds=rnd, kept=kept, rolled_back=rolled, errors=errors,
                    elapsed_s=round(elapsed, 1), eval=ev,
                    elo_start=round(prev_elo, 1), elo_end=round(elo, 1), peak_elo=round(peak_elo, 1),
                    checkpoints={"final": str(out / "final.pt"),
                                 "best": str(best_pt) if best_pt.is_file() else None,
                                 "peak_elo": str(out / "peak_elo.pt") if (out / "peak_elo.pt").is_file() else None,
                                 "latest": str(out / "latest.pt")})
    (out / "run.json").write_text(json.dumps(run_meta, indent=2) + "\n")
    progress.close()

    print(f"\ncandidate(s) in {out}")
    print(f"  full eval:  python3 scripts/eval_bot.py --checkpoint {out / 'best.pt'} --vs-production")
    print(f"  promote  :  python3 scripts/promote_model.py {out / 'best.pt'}   # only if it earns it")
    print("  production + registry are unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

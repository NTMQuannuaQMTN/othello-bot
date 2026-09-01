#!/usr/bin/env python3
"""Long, unattended training loop: play Egaroucid, fine-tune on the games, repeat.

    python3 scripts/train_vs_egaroucid.py --hours 8

Each round: play a short match vs **Egaroucid for Console** (GTP), then fine-tune
the model on those games via the project's existing behaviour-cloning path
(`OthelloBot.finetune_from_games` — shaping + an anchored replay buffer + a
**guardrail** that rolls back any update that weakens the bot vs Random).  The
model is loaded once and fine-tuned in place across every round.

Egaroucid's level **ramps up** over the run (``--level-start`` -> ``--level-end``).

Storage is deliberately tiny — **no per-match result files**.  Only:

    checkpoints/experiments/egaroucid_train_<stamp>/
      latest.pt          most recent kept model              (overwritten)
      best.pt  best.json  best by smoothed win% vs Random     (overwritten)
      snapshots/hNN.pt    one per elapsed hour
      final.pt           the model when the run ends
      progress.jsonl     one compact numeric row per round
      run.json           config + live status + final eval

Stop early: ``Ctrl-C``, or ``touch <out>/STOP``.  Nothing here ever writes to
`checkpoints/production/` or `checkpoints/registry.json`; the result is a
*candidate* — evaluate / promote it afterwards with
`scripts/eval_bot.py` / `scripts/promote_model.py`.
"""
from __future__ import annotations

import argparse
import json
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
from othello_rl.evaluation.tournament import play_match  # noqa: E402
from othello_rl.rl.checkpoint import Registry, resolve_checkpoint  # noqa: E402
from othello_rl.utils.seed import seed_everything  # noqa: E402
from othello_rl.webapp.bot_service import FineTuneConfig, OthelloBot  # noqa: E402


def _parse_duration(hours, minutes, seconds) -> float:
    total = (hours or 0) * 3600.0 + (minutes or 0) * 60.0 + (seconds or 0)
    return total if total > 0 else 8 * 3600.0


def _ramp_level(elapsed: float, duration: float, lo: int, hi: int) -> int:
    if hi == lo or duration <= 0:
        return lo
    frac = min(1.0, max(0.0, elapsed / duration))
    return int(round(lo + (hi - lo) * frac))


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
    ap.add_argument("--games", type=int, default=10, help="games per match (default 10)")
    ap.add_argument("--level-start", type=int, default=1)
    ap.add_argument("--level-end", type=int, default=8)
    ap.add_argument("--opening-plies", type=int, default=4)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--egaroucid", default=None, help="path to Egaroucid_for_Console.out")
    ap.add_argument("--grad-steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--grade-lookahead", type=int, default=1,
                    help="negamax depth for grading moves (shaping signal). The "
                         "big per-round cost — 1 is fast, 3 (the app default) is ~4x slower")
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

    # -- continue round numbering / counters when resuming --------------
    prev_rounds = prev_kept = prev_rolled = 0
    prog_path = out / "progress.jsonl"
    if args.resume and prog_path.is_file():
        for line in prog_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            prev_rounds = max(prev_rounds, int(r.get("round", 0)))
            if isinstance(r.get("ft"), dict):
                prev_kept += bool(r["ft"].get("kept"))
                prev_rolled += (not r["ft"].get("kept"))
        print(f"  resuming after round {prev_rounds} "
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

    print(f"train_vs_egaroucid — {_fmt(duration)} from now")
    print(f"  base model : {base_version}  [{base_label}]  ({n_params:,} params)")
    print(f"  match      : {args.games} games/round, Egaroucid level "
          f"{args.level_start} -> {args.level_end} (ramped), {args.opening_plies} opening plies")
    print(f"  fine-tune  : {ft.grad_steps} grad steps, grade-lookahead {args.grade_lookahead}, "
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
    rnd = prev_rounds
    kept = prev_kept
    rolled = prev_rolled
    errors = 0
    ema_wr = None
    best_wr = prev_best
    hours_saved = set()
    slept_s = 0.0

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
            now_mono = time.monotonic()
            # a big wall gap with almost no monotonic movement == the Mac slept
            gap = (time.time() - w0) - (now_mono - t0)
            if gap - slept_s > 120:
                just_slept = gap - slept_s
                slept_s = gap
                print(f"[{_fmt(now_mono - t0)}] (was asleep ~{_fmt(just_slept)} — "
                      f"compute clock paused; run under `caffeinate` to avoid this)")
            elapsed = now_mono - t0
            level = _ramp_level(elapsed, duration, args.level_start, args.level_end)
            if level != cur_level:
                _open_engine(level)
                print(f"[{_fmt(elapsed)}] round {rnd}: Egaroucid level -> {level}")

            row = {"round": rnd, "t": round(elapsed, 1), "level": level}
            try:
                summ = run_match(bot, engine, games=args.games, opening_plies=args.opening_plies,
                                 seed=args.seed + rnd * 7919, verbose=False)
                row["match"] = {"rl_w": summ.rl_wins, "eg_w": summ.egaroucid_wins,
                                "draw": summ.draws, "disc_diff": round(summ.mean_disc_diff, 1),
                                "rl_score": round(summ.rl_mean_score, 1)}
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
                                          grade_lookahead=args.grade_lookahead)
            except Exception as exc:
                errors += 1
                row["error"] = f"finetune: {exc}"
                progress.write(json.dumps(row) + "\n")
                print(f"[{_fmt(elapsed)}] round {rnd}: finetune error ({exc})")
                continue

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

                h = int(elapsed // 3600)
                if h >= 1 and h not in hours_saved:
                    hours_saved.add(h)
                    _save(out / "snapshots" / f"h{h:02d}.pt")

                if args.anchor_refill_every and rnd % args.anchor_refill_every == 0:
                    bot._ensure_buffer()
                    bot._fill_anchor(args.anchor_refill)   # top the baseline games back up
            except Exception as exc:
                errors += 1
                row["bookkeeping_error"] = str(exc)
                print(f"[{_fmt(elapsed)}] round {rnd}: post-finetune error ({exc}); continuing")

            row["best"] = is_best
            progress.write(json.dumps(row) + "\n")
            eta = _fmt(max(0.0, deadline - time.monotonic()))
            if rnd % 5 == 0 or is_best or "check" in row:
                m = row.get("match", {})
                chk = f" check R/G {row['check']['random']:.2f}/{row['check']['greedy']:.2f}" \
                    if "check" in row else ""
                print(f"[{_fmt(elapsed)}] round {rnd:>4} L{level} | "
                      f"match {m.get('rl_w','?')}-{m.get('eg_w','?')} diff {m.get('disc_diff','?'):>6} | "
                      f"ft {'kept ' if was_kept else 'ROLL '}"
                      f"wr_rand {row['ft']['wr_rand'][0]:.2f}->{wr:.2f} ema {ema_wr:.3f}{chk}"
                      f"{'  <- BEST' if is_best else ''} | kept {kept}/{rnd} | ETA {eta}")

            run_meta.update(status="running", rounds=rnd, kept=kept, rolled_back=rolled,
                            errors=errors, best_score=round(best_wr, 3),
                            wr_rand_ema=round(ema_wr, 3), current_level=level,
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
    elapsed = time.monotonic() - t0
    new_rounds = rnd - prev_rounds
    print(f"\n{'=' * 60}\ntraining loop ended — {_fmt(elapsed)} compute"
          f"{f' (+{_fmt(slept_s)} asleep)' if slept_s > 60 else ''}, "
          f"{new_rounds} rounds this run / {rnd} total, "
          f"{kept} kept, {rolled} rolled back, {errors} errors")

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
                    checkpoints={"final": str(out / "final.pt"),
                                 "best": str(best_pt) if best_pt.is_file() else None,
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

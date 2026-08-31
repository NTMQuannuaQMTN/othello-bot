#!/usr/bin/env python3
"""Orchestrate the historical-data pipeline end to end:

    ingest -> validate -> analyze -> dataset -> pretrain -> eval -> promote

Each stage runs as a subprocess with its own config (configs/iterate.yaml points
at them). One row per stage is written to experiments/index.jsonl so the run is
auditable. Use --from / --to to run a subset, --dry-run to just print the plan.

    python3 scripts/iterate.py --dry-run
    python3 scripts/iterate.py --from analyze --to promote

The RL half (AZ-style MCTS self-play feeding new games back in) is deferred to
docs/alphazero-plan.md.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from othello_rl.utils.config import load_config  # noqa: E402
from othello_rl.utils.experiment import log_experiment  # noqa: E402

STAGES = ["ingest", "validate", "analyze", "dataset", "pretrain", "eval", "promote"]


def _latest_dataset() -> str:
    root = _ROOT / "data" / "processed" / "training_data"
    dirs = sorted((d for d in root.glob("*") if (d / "train.npz").is_file()),
                  key=lambda d: d.stat().st_mtime)
    return dirs[-1].name if dirs else "<no dataset>"


def _latest_candidate() -> str:
    exp = _ROOT / "checkpoints" / "experiments"
    pts = sorted(exp.glob("*_pretrain.pt"), key=lambda p: p.stat().st_mtime)
    return str(pts[-1].relative_to(_ROOT)) if pts else "<no candidate>"


def _cmd(stage: str, cfg: dict) -> list:
    src = cfg.get("source", "wthor")
    py = [sys.executable]
    if stage == "ingest":
        c = [*py, "scripts/ingest_games.py", "--source", src]
        if (cfg.get("ingest") or {}).get("limit"):
            c += ["--limit", str(cfg["ingest"]["limit"])]
        return c
    if stage == "validate":
        return [*py, "scripts/validate_games.py", "--source", src]
    if stage == "analyze":
        a = cfg.get("analyze") or {}
        c = [*py, "scripts/analyze_games.py", "--source", src,
             "--config", a.get("config", "configs/analysis.yaml")]
        if a.get("limit"):
            c += ["--limit", str(a["limit"])]
        return c
    if stage == "dataset":
        d = cfg.get("dataset") or {}
        return [*py, "scripts/build_dataset.py", "--config", d.get("config", "configs/dataset.yaml")]
    if stage == "pretrain":
        p = cfg.get("pretrain") or {}
        return [*py, "scripts/pretrain.py", "--config", p.get("config", "configs/pretrain.yaml"),
                "--dataset", cfg["_dataset_version"]]
    if stage == "eval":
        e = cfg.get("eval") or {}
        c = [*py, "scripts/eval_bot.py", "--checkpoint", cfg["_candidate"],
             "--games", str(e.get("games", 200))]
        if e.get("vs_production"):
            c += ["--vs-production"]
        return c
    if stage == "promote":
        pr = cfg.get("promote") or {}
        return [*py, "scripts/promote_model.py", cfg["_candidate"],
                "--config", pr.get("config", "configs/pretrain.yaml"),
                "--games", str(pr.get("games", 200))]
    raise ValueError(stage)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(_ROOT / "configs" / "iterate.yaml"))
    ap.add_argument("--from", dest="frm", choices=STAGES, default=None)
    ap.add_argument("--to", dest="to", choices=STAGES, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    cfg = dict(load_config(args.config))
    want = cfg.get("stages", STAGES)
    i0 = STAGES.index(args.frm) if args.frm else 0
    i1 = STAGES.index(args.to) if args.to else len(STAGES) - 1
    plan = [s for s in STAGES[i0:i1 + 1] if s in want]

    # later stages consume the newest artifact from the earlier ones
    cfg["_dataset_version"] = "<built by dataset stage>" if args.dry_run else _latest_dataset()
    cfg["_candidate"] = "<written by pretrain stage>" if args.dry_run else _latest_candidate()

    print("plan:", " -> ".join(plan))
    for stage in plan:
        # refresh artifact pointers so a stage sees what the previous one produced
        if not args.dry_run:
            cfg["_dataset_version"] = _latest_dataset()
            cfg["_candidate"] = _latest_candidate()
        cmd = _cmd(stage, cfg)
        print(f"\n$ {' '.join(cmd)}")
        if args.dry_run:
            continue
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=_ROOT).returncode
        dt = round(time.time() - t0, 1)
        log_experiment({"kind": "iterate", "stage": stage, "status": "ok" if rc == 0 else "fail",
                        "returncode": rc, "seconds": dt})
        if rc != 0:
            print(f"stage {stage} failed (rc={rc}) — stopping")
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

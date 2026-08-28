#!/usr/bin/env python3
"""Train the DQN agent through a fixed-opponent curriculum.

Usage:
    python3 scripts/train.py --config configs/train.yaml [--steps-scale 0.1] [--stages stage1_random]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from othello_rl.rl.agent import DQNAgent, NetworkConfig  # noqa: E402
from othello_rl.rl.curriculum import CurriculumConfig, Stage, run_curriculum  # noqa: E402
from othello_rl.rl.trainer import DQNConfig  # noqa: E402
from othello_rl.utils.config import dump_config, load_config  # noqa: E402
from othello_rl.utils.experiment import create_run_dir, write_metadata  # noqa: E402
from othello_rl.utils.logging import MetricLogger  # noqa: E402
from othello_rl.utils.plots import line_plot, winrate_vs_steps  # noqa: E402
from othello_rl.utils.seed import seed_everything  # noqa: E402


def build_curriculum(cfg, steps_scale: float, only_stages) -> CurriculumConfig:
    stages = []
    for s in cfg["stages"]:
        if only_stages and s["name"] not in only_stages:
            continue
        stages.append(Stage(
            name=s["name"], opponent=s["opponent"],
            env_steps=max(1, int(s["env_steps"] * steps_scale)),
            learner_color=s.get("learner_color", "random"),
        ))
    ev = cfg.get("eval", {})
    return CurriculumConfig(
        stages=stages,
        eval_opponents=dict(ev.get("opponents", {"random": "random"})),
        eval_games=int(ev.get("games", 100)),
        eval_every=max(1, int(int(ev.get("every", 5000)) * steps_scale)),
        eval_seed=int(ev.get("seed", 12345)),
        checkpoint_every=max(1, int(int(cfg.get("checkpoint_every", 20000)) * steps_scale)),
        dqn=DQNConfig(**cfg.get("dqn", {})),
    )


def make_plots(run_dir: Path, opponents):
    rows = MetricLogger.load(run_dir / "metrics.jsonl")
    eval_rows = [r for r in rows if r.get("phase") == "eval"]
    if eval_rows:
        winrate_vs_steps(eval_rows, list(opponents), run_dir / "winrate_vs_steps.png",
                         title="DQN win rate vs fixed opponents")
    train_rows = [r for r in rows if r.get("phase") == "train" and "mean_return_100" in r]
    if train_rows:
        xs = [r["env_steps"] for r in train_rows]
        line_plot({"mean return (100 ep)": (xs, [r["mean_return_100"] for r in train_rows])},
                  run_dir / "train_return.png", xlabel="env_steps", ylabel="mean return",
                  title="Training return", ylim=(-1.05, 1.05), hlines=[0.0])


def summarise(run_dir: Path, opponents):
    rows = MetricLogger.load(run_dir / "metrics.jsonl")
    eval_rows = [r for r in rows if r.get("phase") == "eval"]
    lines = ["# Training summary", ""]
    if eval_rows:
        first, last = eval_rows[0], eval_rows[-1]
        lines += ["| opponent | untrained win rate | final win rate |", "|---|---:|---:|"]
        for opp in opponents:
            k = f"winrate_vs_{opp}"
            lines.append(f"| {opp} | {first.get(k, float('nan')):.3f} | {last.get(k, float('nan')):.3f} |")
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--steps-scale", type=float, default=1.0,
                    help="multiply every stage's env_steps (for quick smoke runs)")
    ap.add_argument("--stages", nargs="*", default=None, help="only run these stage names")
    ap.add_argument("--out", default="experiments")
    ap.add_argument("--progress", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 0))
    seed_everything(seed)

    run_dir = create_run_dir(args.out, cfg.get("tag", "train"))
    dump_config(dict(cfg), run_dir / "resolved_config.yaml")

    net_cfg = NetworkConfig(**cfg.get("network", {}))
    agent = DQNAgent(net_cfg, device=str(cfg.get("device", "cpu")), seed=seed)

    cur = build_curriculum(cfg, args.steps_scale, set(args.stages) if args.stages else None)
    write_metadata(run_dir, dict(cfg), extra={"steps_scale": args.steps_scale,
                                              "stages_run": [s.name for s in cur.stages]})
    print(f"run dir: {run_dir}")
    print(f"stages: {[(s.name, s.env_steps) for s in cur.stages]}")

    run_curriculum(agent, cur, run_dir, seed=seed, progress=args.progress)

    opponents = list(cur.eval_opponents)
    make_plots(run_dir, opponents)
    summarise(run_dir, opponents)
    print(f"\nartifacts in {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

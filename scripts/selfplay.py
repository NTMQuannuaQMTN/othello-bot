#!/usr/bin/env python3
"""Self-play training with a mixed opponent pool (Phase 7).

Usage:
    python3 scripts/selfplay.py --config configs/selfplay.yaml \
        [--init experiments/<run>/checkpoints/final.pt] [--steps-scale 0.1]
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
from othello_rl.rl.self_play import OpponentPool, SelfPlayConfig, run_self_play  # noqa: E402
from othello_rl.rl.trainer import DQNConfig  # noqa: E402
from othello_rl.utils.config import dump_config, load_config  # noqa: E402
from othello_rl.utils.experiment import create_run_dir, write_metadata  # noqa: E402
from othello_rl.utils.logging import MetricLogger  # noqa: E402
from othello_rl.utils.plots import line_plot  # noqa: E402
from othello_rl.utils.seed import seed_everything  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/selfplay.yaml")
    ap.add_argument("--init", default=None, help="warm-start checkpoint (overrides config)")
    ap.add_argument("--steps-scale", type=float, default=1.0)
    ap.add_argument("--out", default="experiments")
    ap.add_argument("--progress", choices=["auto", "on", "off"], default="auto")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 0))
    seed_everything(seed)

    run_dir = create_run_dir(args.out, cfg.get("tag", "selfplay"))
    dump_config(dict(cfg), run_dir / "resolved_config.yaml")

    init_ckpt = args.init or cfg.get("init_checkpoint")
    device = str(cfg.get("device", "cpu"))
    if init_ckpt:
        agent = DQNAgent.from_checkpoint(init_ckpt, device=device, seed=seed)
        agent.name = "dqn"
        print(f"warm-started from {init_ckpt} (env_steps={agent.meta.env_steps})")
    else:
        agent = DQNAgent(NetworkConfig(**cfg.get("network", {})), device=device, seed=seed)

    sp = cfg.get("self_play", {})
    pool_cfg = dict(sp.get("pool", {}))
    pool_kwargs = dict(
        baseline_specs=tuple(pool_cfg.get("baseline_specs", ["random", "greedy", "heuristic"])),
        recent_capacity=int(pool_cfg.get("recent_capacity", 5)),
        historical_every=int(pool_cfg.get("historical_every", 3)),
        seed=seed,
    )
    if pool_cfg.get("distribution"):
        pool_kwargs["distribution"] = dict(pool_cfg["distribution"])
    pool = OpponentPool(**pool_kwargs)
    scale = args.steps_scale
    sp_cfg = SelfPlayConfig(
        total_env_steps=max(1, int(sp.get("total_env_steps", 200000) * scale)),
        snapshot_every=max(1, int(sp.get("snapshot_every", 20000) * scale)),
        eval_every=max(1, int(sp.get("eval_every", 20000) * scale)),
        eval_games=int(sp.get("eval_games", 100)),
        checkpoint_every=max(1, int(sp.get("checkpoint_every", 40000) * scale)),
        learner_color=str(sp.get("learner_color", "random")),
        opening_plies=int(sp.get("opening_plies", 4)),
        pool=pool,
        dqn=DQNConfig(**cfg.get("dqn", {})),
    )

    write_metadata(run_dir, dict(cfg), extra={"init_checkpoint": init_ckpt,
                                              "steps_scale": scale})
    print(f"run dir: {run_dir}")
    progress = {"auto": "auto", "on": True, "off": False}[args.progress]
    run_self_play(agent, sp_cfg, run_dir, seed=seed, progress=progress)

    # plots
    rows = MetricLogger.load(run_dir / "metrics.jsonl")
    ev = [r for r in rows if r.get("phase") == "eval"]
    if ev:
        xs = [r["env_steps"] for r in ev]
        series = {}
        for key in sorted({k for r in ev for k in r if k.startswith("winrate_vs_")}):
            series[key.replace("winrate_vs_", "")] = (
                [r["env_steps"] for r in ev if key in r],
                [r[key] for r in ev if key in r],
            )
        line_plot(series, run_dir / "selfplay_winrate.png", xlabel="env_steps",
                  ylabel="win rate (draw=0.5)", title="Self-play: win rate vs pool members",
                  hlines=[0.5], ylim=(0, 1))
    print(f"\nartifacts in {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

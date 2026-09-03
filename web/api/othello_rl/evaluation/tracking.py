"""Evaluate a series of checkpoints to track playing strength over training.

Produces, for every checkpoint:
- win rate / disc diff vs each fixed baseline,
- an internal Elo from a round-robin among the checkpoints plus baseline anchors.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .elo import ratings_from_matches
from .harness import evaluate_agent
from .tournament import play_match

_STEP_RE = re.compile(r"(?:step|_)(\d+)")


def _load_checkpoint_agent(path: Path):
    from othello_rl.rl.agent import DQNAgent
    agent = DQNAgent.from_checkpoint(path)
    agent.name = path.stem
    step = agent.meta.env_steps
    if not step:
        m = _STEP_RE.search(path.stem)
        step = int(m.group(1)) if m else 0
    return agent, int(step)


def discover_checkpoints(run_dir: str | Path) -> List[Path]:
    ckpt_dir = Path(run_dir) / "checkpoints"
    if not ckpt_dir.is_dir():
        ckpt_dir = Path(run_dir)
    return sorted(ckpt_dir.glob("*.pt"))


def track_checkpoints(paths: Sequence[Path],
                      baselines: Optional[Dict[str, str]] = None,
                      games: int = 60, seed: int = 4242,
                      opening_plies: int = 4,
                      round_robin_games: int = 40) -> dict:
    baselines = baselines or {"random": "random", "greedy": "greedy",
                              "heuristic": "heuristic"}
    loaded = [_load_checkpoint_agent(Path(p)) for p in paths]
    loaded.sort(key=lambda t: t[1])

    entries: List[dict] = []
    for agent, step in loaded:
        res = evaluate_agent(agent, baselines, num_games=games, seed=seed,
                             opening_plies=opening_plies)
        entries.append({
            "name": agent.name,
            "env_steps": step,
            "vs": {k: {"win_rate": v["win_rate"], "ci_low": v["ci_low"],
                       "ci_high": v["ci_high"], "mean_disc_diff": v["mean_disc_diff"]}
                   for k, v in res.items()},
        })

    # round-robin among checkpoints + baseline anchors for internal Elo
    agents = [a for a, _ in loaded]
    names = [a.name for a in agents]
    matches = []
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            matches.append(play_match(agents[i], agents[j], num_games=round_robin_games,
                                      seed=seed + i * 100 + j, opening_plies=opening_plies))
    for ai, a in enumerate(agents):
        for bi, (bname, bspec) in enumerate(baselines.items()):
            matches.append(play_match(a, bspec, num_games=round_robin_games,
                                      seed=seed + 7919 * (ai + 1) + 31 * (bi + 1),
                                      opening_plies=opening_plies))
    elo = ratings_from_matches(matches, anchor="random")
    ratings = elo.leaderboard()

    for e in entries:
        e["internal_elo"] = elo.rating(e["name"])

    return {
        "elo_kind": elo.kind,
        "checkpoints": entries,
        "elo_leaderboard": [{"name": n, "rating": r} for n, r in ratings],
        "baselines": list(baselines),
    }


def write_tracking_report(result: dict, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tracking.json").write_text(json.dumps(result, indent=2))

    baselines = result["baselines"]
    lines = ["# Checkpoint strength tracking", "",
             f"Internal Elo ({result['elo_kind']}) — comparable only within this run.",
             "",
             "| checkpoint | env_steps | " + " | ".join(f"vs {b}" for b in baselines)
             + " | internal Elo |",
             "|---|--:|" + "--:|" * (len(baselines) + 1)]
    for e in result["checkpoints"]:
        cells = [f"{e['vs'][b]['win_rate']:.3f}" for b in baselines]
        lines.append(f"| {e['name']} | {e['env_steps']} | " + " | ".join(cells)
                     + f" | {e['internal_elo']:.0f} |")
    path = out_dir / "tracking.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def write_tracking_plots(result: dict, out_dir: str | Path):
    from othello_rl.utils.plots import line_plot
    out_dir = Path(out_dir)
    ckpts = [e for e in result["checkpoints"] if e["env_steps"] is not None]
    ckpts.sort(key=lambda e: e["env_steps"])
    xs = [e["env_steps"] for e in ckpts]

    wr_series = {b: (xs, [e["vs"][b]["win_rate"] for e in ckpts]) for b in result["baselines"]}
    line_plot(wr_series, out_dir / "winrate_vs_checkpoint.png",
              xlabel="training env steps", ylabel="win rate (draw=0.5)",
              title="Checkpoint win rate vs baselines", hlines=[0.5], ylim=(0, 1))

    line_plot({"internal Elo": (xs, [e["internal_elo"] for e in ckpts])},
              out_dir / "elo_vs_checkpoint.png",
              xlabel="training env steps", ylabel="internal Elo",
              title="Checkpoint internal Elo")

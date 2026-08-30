"""Standard single-checkpoint evaluation protocol (PROJECT_SPEC Phase 8):
score one agent against a fixed panel — Random, Greedy, Heuristic and Minimax at
several depths — with Wilson confidence intervals, and place it on an internal
Elo scale anchored to the panel.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional

from .elo import ratings_from_matches
from .metrics import summarize_match
from .tournament import AgentSpec, play_match

STANDARD_PANEL: Dict[str, str] = {
    "random": "random",
    "greedy": "greedy",
    "heuristic": "heuristic",
    "minimax:1": "minimax:1",
    "minimax:2": "minimax:2",
    "minimax:3": "minimax:3",
}


def standard_panel_eval(agent: AgentSpec, *, agent_name: str = "agent",
                        panel: Optional[Mapping[str, str]] = None,
                        num_games: int = 100, elo_extra_games: int = 40,
                        seed: int = 20260829, opening_plies: int = 4) -> dict:
    """Evaluate ``agent`` vs ``panel`` (default :data:`STANDARD_PANEL`).

    Runs ``num_games`` per opponent (the expensive part) and reuses those matches
    to fit an internal Elo (``random`` anchored to 1500). ``elo_extra_games``
    adds a light random↔greedy↔heuristic sub-round-robin so the weak end of the
    scale is also constrained (skip with ``elo_extra_games=0``).
    """
    panel = dict(panel or STANDARD_PANEL)

    vs: Dict[str, Dict[str, float]] = {}
    agent_matches = []
    for i, (name, spec) in enumerate(panel.items()):
        m = play_match(agent, spec, num_games=num_games, seed=seed + 1000 * i,
                       opening_plies=opening_plies)
        m.name_a = agent_name
        agent_matches.append(m)
        s = summarize_match(m)
        vs[name] = {
            "win_rate": s.a_win_rate, "ci_low": s.ci_low, "ci_high": s.ci_high,
            "wins": s.a_wins, "losses": s.b_wins, "draws": s.draws,
            "mean_disc_diff": s.mean_disc_diff,
        }

    elo_matches = list(agent_matches)
    if elo_extra_games > 0:
        cheap = [n for n in ("random", "greedy", "heuristic") if n in panel]
        pair_no = 0
        for x in range(len(cheap)):
            for y in range(x + 1, len(cheap)):
                pair_no += 1
                elo_matches.append(play_match(panel[cheap[x]], panel[cheap[y]],
                                              num_games=elo_extra_games,
                                              seed=seed + 555 + pair_no,
                                              opening_plies=opening_plies))
    elo = ratings_from_matches(elo_matches, anchor="random")

    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "agent": agent_name,
        "num_games": num_games,
        "elo_extra_games": elo_extra_games,
        "seed": seed,
        "panel": list(panel),
        "vs": vs,
        "elo_kind": elo.kind,
        "internal_elo": dict(elo.leaderboard()),
        "agent_internal_elo": elo.rating(agent_name),
    }


def write_bot_report(result: dict, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bot_eval.json").write_text(json.dumps(result, indent=2))

    lines: List[str] = [
        f"# Standard evaluation — {result['agent']}",
        "",
        f"_{result['generated_utc']} · {result['num_games']} games/opponent · "
        f"seed {result['seed']} · random openings_",
        "",
        "| opponent | W–L–D | win rate | 95% CI | mean disc diff | sig.? |",
        "|---|---|--:|:--:|--:|:--:|",
    ]
    for name in result["panel"]:
        d = result["vs"][name]
        sig = "yes" if (d["ci_low"] > 0.5 or d["ci_high"] < 0.5) else "no"
        lines.append(
            f"| {name} | {d['wins']}–{d['losses']}–{d['draws']} | {d['win_rate']:.3f} | "
            f"[{d['ci_low']:.2f}, {d['ci_high']:.2f}] | {d['mean_disc_diff']:+.1f} | {sig} |"
        )
    lines += [
        "",
        f"## Internal Elo ({result['elo_kind']}, random = 1500)",
        "",
        "> Comparable only within this panel. Not an external rating.",
        "",
        "| agent | rating |",
        "|---|--:|",
    ]
    for name, rating in sorted(result["internal_elo"].items(), key=lambda kv: -kv[1]):
        mark = " **(this bot)**" if name == result["agent"] else ""
        lines.append(f"| {name}{mark} | {rating:.0f} |")
    lines.append("")
    path = out_dir / "bot_eval.md"
    path.write_text("\n".join(lines))
    return path

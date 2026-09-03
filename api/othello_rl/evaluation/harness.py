"""High-level evaluation: score one agent against a set of named opponents."""
from __future__ import annotations

from typing import Dict, Mapping

from .metrics import summarize_match
from .tournament import AgentSpec, play_match


def evaluate_agent(agent: AgentSpec, opponents: Mapping[str, AgentSpec],
                   num_games: int = 100, seed: int = 0, opening_plies: int = 4
                   ) -> Dict[str, Dict[str, float]]:
    """Return ``{opponent_name: {win_rate, ci_low, ci_high, wins, losses, draws,
    mean_disc_diff}}`` for ``agent`` (stats from the agent's perspective)."""
    out: Dict[str, Dict[str, float]] = {}
    for i, (name, opp) in enumerate(opponents.items()):
        m = play_match(agent, opp, num_games=num_games, seed=seed + 1000 * i,
                       opening_plies=opening_plies)
        s = summarize_match(m)
        out[name] = {
            "win_rate": s.a_win_rate,
            "ci_low": s.ci_low,
            "ci_high": s.ci_high,
            "wins": s.a_wins,
            "losses": s.b_wins,
            "draws": s.draws,
            "mean_disc_diff": s.mean_disc_diff,
        }
    return out


def flatten_eval(result: Mapping[str, Mapping[str, float]], prefix: str = "winrate_vs_"
                 ) -> Dict[str, float]:
    """Flatten :func:`evaluate_agent` output to ``{winrate_vs_<name>: rate}`` for
    metric logging."""
    flat: Dict[str, float] = {}
    for name, d in result.items():
        flat[f"{prefix}{name}"] = d["win_rate"]
        flat[f"disc_diff_vs_{name}"] = d["mean_disc_diff"]
    return flat

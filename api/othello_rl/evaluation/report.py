"""Turn match results into a machine-readable JSON blob, a markdown report and a
simple bar chart."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from .elo import EloModel, ratings_from_matches
from .metrics import MatchSummary, summarize_match


def build_results(matches, seed: int, extra: Optional[dict] = None) -> dict:
    summaries: List[MatchSummary] = [summarize_match(m) for m in matches]
    elo: EloModel = ratings_from_matches(matches)
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "elo_kind": elo.kind,
        "elo": dict(elo.leaderboard()),
        "matches": [s.to_dict() for s in summaries],
        **(extra or {}),
    }


def write_json(results: dict, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "results.json"
    path.write_text(json.dumps(results, indent=2))
    return path


def write_markdown(results: dict, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Baseline evaluation report",
        "",
        f"_Generated {results['generated_utc']} · seed {results['seed']}_",
        "",
        "## Matches",
        "",
        "| A | B | games | A wins | B wins | draws | A win rate | 95% CI | mean disc diff | sig.? |",
        "|---|---|------:|------:|------:|-----:|----------:|:------:|--------------:|:---:|",
    ]
    for m in results["matches"]:
        ci = f"[{m['ci_low']:.2f}, {m['ci_high']:.2f}]"
        sig = "yes" if (m["ci_low"] > 0.5 or m["ci_high"] < 0.5) else "no"
        lines.append(
            f"| {m['name_a']} | {m['name_b']} | {m['num_games']} | {m['a_wins']} | "
            f"{m['b_wins']} | {m['draws']} | {m['a_win_rate']:.3f} | {ci} | "
            f"{m['mean_disc_diff']:+.2f} | {sig} |"
        )
    lines += ["", f"## Internal Elo ({results['elo_kind']})", "",
              "> Internal experimental Elo — comparable only within this project. "
              "Not an external/online rating.", "",
              "| agent | rating |", "|---|---:|"]
    for name, rating in sorted(results["elo"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {name} | {rating:.0f} |")
    lines.append("")
    path = out_dir / "report.md"
    path.write_text("\n".join(lines))
    return path


def write_plot(results: dict, out_dir: Path) -> Optional[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [f"{m['name_a']}\nvs\n{m['name_b']}" for m in results["matches"]]
    rates = [m["a_win_rate"] for m in results["matches"]]
    errs_low = [max(0.0, m["a_win_rate"] - m["ci_low"]) for m in results["matches"]]
    errs_high = [max(0.0, m["ci_high"] - m["a_win_rate"]) for m in results["matches"]]
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(labels)), 4))
    ax.bar(range(len(labels)), rates, yerr=[errs_low, errs_high], capsize=4,
           color="#4C72B0")
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Agent A win rate (draws = 0.5)")
    ax.set_title("Baseline matchups")
    fig.tight_layout()
    path = out_dir / "matchups.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def generate_report(matches, out_dir, seed: int, extra: Optional[dict] = None) -> dict:
    results = build_results(matches, seed=seed, extra=extra)
    write_json(results, out_dir)
    write_markdown(results, out_dir)
    write_plot(results, out_dir)
    return results

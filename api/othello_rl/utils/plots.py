"""Matplotlib helpers for training/evaluation curves. Import is lazy so the rest
of the package works without matplotlib installed."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def line_plot(series: Dict[str, Tuple[Sequence[float], Sequence[float]]],
              out_path: str | Path, *, xlabel: str, ylabel: str, title: str,
              hlines: Optional[Iterable[float]] = None, ylim: Optional[Tuple[float, float]] = None) -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, (xs, ys) in series.items():
        ax.plot(xs, ys, marker="o", markersize=3, label=label)
    for h in (hlines or []):
        ax.axhline(h, color="grey", linestyle="--", linewidth=1)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if len(series) > 1 or True:
        ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def winrate_vs_steps(rows: List[dict], opponents: Sequence[str], out_path: str | Path,
                     x: str = "env_steps", title: str = "Win rate vs baselines") -> Path:
    series = {}
    for opp in opponents:
        key = f"winrate_vs_{opp}"
        pts = [(r[x], r[key]) for r in rows if x in r and key in r]
        if pts:
            series[opp] = ([p[0] for p in pts], [p[1] for p in pts])
    return line_plot(series, out_path, xlabel=x, ylabel="win rate (draw=0.5)",
                     title=title, hlines=[0.5], ylim=(0.0, 1.0))

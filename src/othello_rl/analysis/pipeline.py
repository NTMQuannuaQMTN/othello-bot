"""Analyse VALID games -> per-move quality labels. Games are independent, so an
optional process pool parallelises. Benchmark before a full run."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from othello_rl.agents.heuristic_agent import DEFAULT_WEIGHTS
from othello_rl.ingest.records import GameRecord

from .counterfactual import DEFAULT_MOVE_QUALITY, DEFAULT_SCALE, judge_position
from .reconstruct import positions

DEFAULT_CONFIG = {
    "lookahead_plies": [3, 5],
    "alpha_beta": True,
    "transposition_table": True,
    "max_alternatives": 0,          # 0 => every legal move
    "evaluation_scale": DEFAULT_SCALE,
    "move_quality": dict(DEFAULT_MOVE_QUALITY),
    "n_workers": 1,
}


def _cfg(user: Optional[dict]) -> dict:
    c = {**DEFAULT_CONFIG, **(user or {})}
    c["move_quality"] = {**DEFAULT_MOVE_QUALITY, **(c.get("move_quality") or {})}
    return c


def analyze_game(record: GameRecord, cfg: Optional[dict] = None) -> dict:
    c = _cfg(cfg)
    weights = DEFAULT_WEIGHTS
    moves: List[dict] = []
    for pos in positions(record):
        by_h = {}
        for h in c["lookahead_plies"]:
            j = judge_position(
                pos, int(h), weights=weights, move_quality=c["move_quality"],
                scale=float(c["evaluation_scale"]), alpha_beta=bool(c["alpha_beta"]),
                transposition=bool(c["transposition_table"]),
                max_alternatives=int(c["max_alternatives"]))
            by_h[str(h)] = j.as_dict()
        moves.append({"move_number": pos.move_number, "player": pos.side,
                      "played_move": pos.played_move, "n_legal": len(pos.legal_moves),
                      "by_horizon": by_h})
    return {"game_id": record.game_id, "source": record.source,
            "data_kind": record.data_kind, "result": record.result,
            "n_positions": len(moves), "moves": moves}


def _worker(args):
    rec_json, cfg = args
    return analyze_game(GameRecord.from_json(rec_json), cfg)


@dataclass
class AnalysisStats:
    games: int = 0
    positions: int = 0
    label_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)  # horizon -> label -> n
    seconds: float = 0.0

    def as_dict(self) -> dict:
        d = {"games": self.games, "positions": self.positions,
             "seconds": round(self.seconds, 2),
             "positions_per_sec": round(self.positions / self.seconds, 1) if self.seconds else 0.0,
             "label_counts": self.label_counts}
        return d


def analyze_file(validated_jsonl, out_jsonl, cfg: Optional[dict] = None, *,
                 limit: Optional[int] = None, progress=None,
                 report_dir: Optional[Path] = None) -> AnalysisStats:
    c = _cfg(cfg)
    validated_jsonl, out_jsonl = Path(validated_jsonl), Path(out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    lines = [ln for ln in validated_jsonl.read_text().splitlines() if ln.strip()]
    if limit:
        lines = lines[:limit]

    stats = AnalysisStats()
    t0 = time.time()
    n_workers = max(1, int(c["n_workers"]))

    with out_jsonl.open("w") as fh:
        if n_workers > 1:
            import multiprocessing as mp
            with mp.Pool(n_workers) as pool:
                it = pool.imap(_worker, ((ln, c) for ln in lines), chunksize=8)
                results = _drain(it, fh, stats, c, progress, len(lines))
        else:
            results = _drain((analyze_game(GameRecord.from_json(ln), c) for ln in lines),
                             fh, stats, c, progress, len(lines))
    stats.seconds = time.time() - t0

    payload = {**stats.as_dict(), "config": c, "input": str(validated_jsonl),
               "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    out_jsonl.with_suffix(".stats.json").write_text(json.dumps(payload, indent=2) + "\n")
    if report_dir is not None:
        Path(report_dir).mkdir(parents=True, exist_ok=True)
        (Path(report_dir) / "analysis.stats.json").write_text(json.dumps(payload, indent=2) + "\n")
    return stats


def _drain(results, fh, stats: AnalysisStats, cfg, progress, total) -> None:
    for i, g in enumerate(results):
        fh.write(json.dumps(g, separators=(",", ":")) + "\n")
        stats.games += 1
        stats.positions += g["n_positions"]
        for mv in g["moves"]:
            for h, j in mv["by_horizon"].items():
                stats.label_counts.setdefault(h, {})
                stats.label_counts[h][j["label"]] = stats.label_counts[h].get(j["label"], 0) + 1
        if progress:
            progress(i + 1, total)


# --------------------------------------------------------------------------- #
def benchmark(records: Iterable[GameRecord], cfg: Optional[dict] = None) -> dict:
    """Time analysis on a sample; report positions/sec, branching, @3 vs @5."""
    c = _cfg(cfg)
    recs = list(records)
    n_pos = 0
    branch = 0
    for r in recs:
        for p in positions(r):
            n_pos += 1
            branch += len(p.legal_moves)
    out = {"games": len(recs), "positions": n_pos,
           "mean_branching_factor": round(branch / n_pos, 2) if n_pos else 0.0}
    for h in c["lookahead_plies"]:
        t0 = time.time()
        for r in recs:
            for p in positions(r):
                judge_position(p, int(h), move_quality=c["move_quality"],
                               scale=float(c["evaluation_scale"]),
                               alpha_beta=bool(c["alpha_beta"]),
                               transposition=bool(c["transposition_table"]),
                               max_alternatives=int(c["max_alternatives"]))
        dt = time.time() - t0
        out[f"horizon_{h}"] = {"seconds": round(dt, 2),
                               "positions_per_sec": round(n_pos / dt, 1) if dt else 0.0}
    return out

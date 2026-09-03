"""Run validation over an ingested JSONL: VALID -> validated_games, everything
else -> data/rejected/ with a reason. Stats are written next to the output and
copied into experiments/ (committed)."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from othello_rl.ingest.records import GameRecord

from .replay import Status, validate
from .stats import ValidationStats


def validate_file(in_jsonl, valid_out, rejected_out, *, source: str,
                  report_dir: Optional[Path] = None) -> ValidationStats:
    in_jsonl, valid_out, rejected_out = Path(in_jsonl), Path(valid_out), Path(rejected_out)
    valid_out.parent.mkdir(parents=True, exist_ok=True)
    rejected_out.parent.mkdir(parents=True, exist_ok=True)
    stats = ValidationStats(source=source)

    with valid_out.open("w") as vf, rejected_out.open("w") as rf:
        for line in in_jsonl.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rec = GameRecord.from_json(line)
            res = validate(rec)
            stats.record(res)
            if res.status is Status.VALID:
                rec.canonical_moves = res.canonical_moves
                if rec.result is None:
                    rec.result = {"black_discs": res.final_black,
                                  "white_discs": res.final_white,
                                  "winner": res.replayed_winner}
                rec.provenance = {**rec.provenance,
                                  "validation": {"plies": res.plies_replayed,
                                                 "winner_matches": res.winner_matches}}
                vf.write(rec.to_json() + "\n")
            else:
                rf.write(json.dumps({"game_id": rec.game_id, "source": rec.source,
                                     "status": res.status.value, "reason": res.reason,
                                     "plies_replayed": res.plies_replayed}) + "\n")

    payload = {**stats.as_dict(), "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "input": str(in_jsonl)}
    (valid_out.with_suffix(".stats.json")).write_text(json.dumps(payload, indent=2) + "\n")
    if report_dir is not None:
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "validation.stats.json").write_text(json.dumps(payload, indent=2) + "\n")
    return stats

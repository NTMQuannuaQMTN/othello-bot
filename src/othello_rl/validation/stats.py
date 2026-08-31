"""Aggregate validation counts. Numbers come only from real runs."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict

from .replay import Status, ValidationResult


@dataclass
class ValidationStats:
    source: str
    total: int = 0
    valid: int = 0
    invalid: int = 0
    incomplete: int = 0
    unsupported: int = 0
    winner_mismatch: int = 0        # VALID games where recorded winner != replayed
    reasons: Dict[str, int] = field(default_factory=dict)

    _STATUS_FIELD = {
        Status.VALID: "valid", Status.INVALID: "invalid",
        Status.INCOMPLETE: "incomplete", Status.UNSUPPORTED_FORMAT: "unsupported",
    }

    def record(self, res: ValidationResult) -> None:
        self.total += 1
        setattr(self, self._STATUS_FIELD[res.status],
                getattr(self, self._STATUS_FIELD[res.status]) + 1)
        if res.status is Status.VALID and res.winner_matches is False:
            self.winner_mismatch += 1
        if res.reason:
            key = res.reason.split(" at ply")[0].split(" #")[0]
            self.reasons[key] = self.reasons.get(key, 0) + 1

    def as_dict(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if not k.startswith("_")}
        d["valid_fraction"] = round(self.valid / self.total, 4) if self.total else 0.0
        return d

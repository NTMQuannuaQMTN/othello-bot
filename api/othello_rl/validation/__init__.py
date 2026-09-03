"""Validate ingested games by replaying every move through the project's own
Othello engine. External game data is never trusted.
"""
from .replay import Status, ValidationResult, validate
from .stats import ValidationStats
from .pipeline import validate_file

__all__ = ["Status", "ValidationResult", "validate", "ValidationStats",
           "validate_file"]

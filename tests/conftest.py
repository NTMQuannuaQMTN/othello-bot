import os

import pytest


@pytest.fixture(autouse=True)
def _isolated_experiment_index(tmp_path_factory, monkeypatch):
    """Never let a test (incl. subprocess script tests) write to the real
    committed experiments/index.jsonl."""
    p = tmp_path_factory.mktemp("expidx") / "index.jsonl"
    monkeypatch.setenv("OTHELLO_EXPERIMENT_INDEX", str(p))

import pytest

from othello_rl.webapp import bot_service


@pytest.fixture(autouse=True)
def _engine_off(monkeypatch):
    """The webapp tests exercise the analysis/serve plumbing, not the search
    engine — run every ``OthelloBot`` with the engine off (raw policy) so they
    stay fast. Dedicated engine tests live in ``tests/engine/``."""
    monkeypatch.setattr(bot_service, "_DEFAULT_ENGINE_BUDGET", 0.0)

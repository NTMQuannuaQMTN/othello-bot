"""Reinforcement-learning components: network, replay, DQN agent, trainer.

Submodules are imported **lazily** (PEP 562) so a torch-free consumer — the web
deploy loads :mod:`othello_rl.rl.numpy_policy` and :mod:`othello_rl.rl.replay_buffer`
only — never pays for ``import torch`` via this package's ``__init__``.
"""
from importlib import import_module

_EXPORTS = {
    "SmallOthelloNet": "network",
    "greedy_action": "network",
    "masked_q": "network",
    "ReplayBuffer": "replay_buffer",
    "Batch": "replay_buffer",
    "DQNAgent": "agent",
    "NetworkConfig": "agent",
    "AgentMeta": "agent",
    "FixedOpponentEnv": "opponents",
    "DQNTrainer": "trainer",
    "DQNConfig": "trainer",
    "TrainMetrics": "trainer",
    "run_curriculum": "curriculum",
    "CurriculumConfig": "curriculum",
    "Stage": "curriculum",
    "OpponentPool": "self_play",
    "SelfPlayConfig": "self_play",
    "run_self_play": "self_play",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    mod = _EXPORTS.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f"{__name__}.{mod}"), name)


def __dir__():
    return sorted(list(globals()) + __all__)

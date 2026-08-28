"""Baseline (non-RL) Othello agents."""
from .base import Agent, Move
from .random_agent import RandomAgent
from .greedy_agent import GreedyAgent
from .heuristic_agent import DEFAULT_WEIGHTS, HeuristicAgent, evaluate
from .minimax_agent import MinimaxAgent

__all__ = [
    "Agent",
    "Move",
    "RandomAgent",
    "GreedyAgent",
    "HeuristicAgent",
    "MinimaxAgent",
    "evaluate",
    "DEFAULT_WEIGHTS",
]


def make_agent(spec: str) -> Agent:
    """Build an agent from a short string spec used in configs/CLIs.

    Examples: ``"random"``, ``"random:7"`` (seed), ``"greedy"``, ``"heuristic"``,
    ``"minimax:3"`` (depth).
    """
    name, _, arg = spec.partition(":")
    name = name.strip().lower()
    if name == "random":
        return RandomAgent(seed=int(arg) if arg else None)
    if name == "greedy":
        return GreedyAgent()
    if name == "heuristic":
        return HeuristicAgent()
    if name == "minimax":
        return MinimaxAgent(depth=int(arg) if arg else 3)
    raise ValueError(f"unknown agent spec: {spec!r}")

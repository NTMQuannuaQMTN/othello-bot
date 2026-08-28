"""Play games and matches between agents, with reproducible seeding.

Terminology
-----------
- **game**: one playthrough; one agent is Black, the other White.
- **match**: ``num_games`` games between two agents, colours alternated so each
  side plays Black half the time. Statistics are reported from *agent A*'s view.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Union

from othello_rl.agents import Agent, make_agent
from othello_rl.environment.board import BLACK, WHITE, Board
from othello_rl.utils.seed import spawn_seed

AgentSpec = Union[str, Agent, Callable[[], Agent]]


@dataclass
class GameResult:
    winner: int          # BLACK / WHITE / 0 (draw)
    black_score: int
    white_score: int
    plies: int
    seed: Optional[int] = None

    @property
    def score_diff(self) -> int:
        """Black discs minus White discs."""
        return self.black_score - self.white_score


def play_game(black: Agent, white: Agent, *, seed: Optional[int] = None,
              max_plies: int = 200, opening_plies: int = 0,
              opening_rng: Optional[random.Random] = None) -> GameResult:
    """Play a single game. Agents must return legal moves (they are trusted).

    ``opening_plies`` random plies are played first (from ``opening_rng``) so that
    matches between deterministic agents still produce a diverse, reproducible
    set of games instead of the same game repeated.
    """
    black.reset()
    white.reset()
    state = Board.initial()
    agents = {BLACK: black, WHITE: white}
    plies = 0
    orng = opening_rng or random.Random(seed)
    while not state.is_terminal():
        if plies < opening_plies:
            moves = state.legal_moves()
            move = orng.choice(moves) if moves else None
        else:
            move = agents[state.player].select_move(state)
        state = state.apply(move)  # Board.apply raises on an illegal move
        plies += 1
        if plies >= max_plies:  # pragma: no cover - safety valve
            raise RuntimeError("game exceeded max_plies; agent likely looping")
    b, w = state.scores()
    return GameResult(state.winner(), b, w, plies, seed)


@dataclass
class MatchResult:
    name_a: str
    name_b: str
    num_games: int
    a_wins: int = 0
    b_wins: int = 0
    draws: int = 0
    a_disc_total: int = 0
    b_disc_total: int = 0
    games: List[GameResult] = field(default_factory=list)

    @property
    def a_score(self) -> float:
        """A's match score (win=1, draw=0.5)."""
        return self.a_wins + 0.5 * self.draws

    @property
    def a_win_rate(self) -> float:
        return self.a_score / self.num_games if self.num_games else 0.0

    @property
    def mean_disc_diff(self) -> float:
        """Mean (A discs - B discs) per game."""
        return (self.a_disc_total - self.b_disc_total) / self.num_games if self.num_games else 0.0

    def to_dict(self) -> dict:
        return {
            "name_a": self.name_a,
            "name_b": self.name_b,
            "num_games": self.num_games,
            "a_wins": self.a_wins,
            "b_wins": self.b_wins,
            "draws": self.draws,
            "a_win_rate": self.a_win_rate,
            "mean_disc_diff": self.mean_disc_diff,
        }


def play_match(agent_a: AgentSpec, agent_b: AgentSpec, num_games: int = 100,
               seed: int = 0, alternate_colors: bool = True,
               opening_plies: int = 4) -> MatchResult:
    """Run ``num_games`` between two agent specs and aggregate from A's view.

    Fresh agent instances are built for every game so per-game seeds make the
    whole match reproducible. ``opening_plies`` random opening moves diversify
    games between deterministic agents (set to 0 to disable).
    """
    master = random.Random(seed)
    name_a = _spec_name(agent_a)
    name_b = _spec_name(agent_b)
    res = MatchResult(name_a, name_b, num_games)

    for g in range(num_games):
        gseed = spawn_seed(master)
        a = make_agent(agent_a, seed=gseed)
        b = make_agent(agent_b, seed=gseed ^ 0x5DEECE66)
        orng = random.Random(gseed ^ 0x9E3779B9)
        a_is_black = (g % 2 == 0) or not alternate_colors
        if a_is_black:
            gr = play_game(a, b, seed=gseed, opening_plies=opening_plies, opening_rng=orng)
            a_discs, b_discs = gr.black_score, gr.white_score
            a_won, b_won = gr.winner == BLACK, gr.winner == WHITE
        else:
            gr = play_game(b, a, seed=gseed, opening_plies=opening_plies, opening_rng=orng)
            a_discs, b_discs = gr.white_score, gr.black_score
            a_won, b_won = gr.winner == WHITE, gr.winner == BLACK

        res.games.append(gr)
        res.a_disc_total += a_discs
        res.b_disc_total += b_discs
        if a_won:
            res.a_wins += 1
        elif b_won:
            res.b_wins += 1
        else:
            res.draws += 1
    return res


def round_robin(specs: List[AgentSpec], num_games: int = 100, seed: int = 0
                ) -> List[MatchResult]:
    """Play every unordered pair once. Seeds are derived per pairing."""
    master = random.Random(seed)
    out: List[MatchResult] = []
    for i in range(len(specs)):
        for j in range(i + 1, len(specs)):
            out.append(play_match(specs[i], specs[j], num_games=num_games,
                                  seed=spawn_seed(master)))
    return out


def _spec_name(spec: AgentSpec) -> str:
    if isinstance(spec, Agent):
        return spec.name
    if callable(spec):
        try:
            return spec().name
        except Exception:  # pragma: no cover
            return getattr(spec, "__name__", "agent")
    return str(spec)

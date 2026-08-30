"""Wrap :class:`OthelloEnv` into a stationary single-agent MDP by having a fixed
opponent play its own moves internally.

The learner always sees observations from its own perspective (the base env is
already canonical for the side to move) and rewards in ``{-1, 0, +1}`` from its
own perspective, non-zero only at episode end.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from othello_rl.agents import Agent, make_agent
from othello_rl.environment.board import BLACK, WHITE, PASS_ACTION
from othello_rl.environment.environment import OthelloEnv

Color = Union[int, str]  # BLACK, WHITE, or "random"


class FixedOpponentEnv:
    def __init__(self, opponent: Union[str, Agent, List], learner_color: Color = "random",
                 illegal_move_mode: str = "raise", seed: Optional[int] = None,
                 opening_plies: int = 0):
        self._opponent_spec = opponent
        self.learner_color = learner_color
        self.opening_plies = int(opening_plies)
        self.env = OthelloEnv(illegal_move_mode=illegal_move_mode)
        self._rng = random.Random(seed)
        self.opponent: Agent = self._make_opponent()
        self.learner_is: int = BLACK

    def _make_opponent(self) -> Agent:
        spec = self._opponent_spec
        if hasattr(spec, "sample"):  # an OpponentPool-like object
            return spec.sample(self._rng)
        if isinstance(spec, list):  # sample from a list of specs each episode
            spec = self._rng.choice(spec)
        return make_agent(spec, seed=self._rng.randrange(2 ** 31))

    # -- API ----------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self._rng.seed(seed)
        self.opponent = self._make_opponent()
        if self.learner_color == "random":
            self.learner_is = self._rng.choice((BLACK, WHITE))
        else:
            self.learner_is = int(self.learner_color)

        for _ in range(20):  # retry if a long random opening ended the game
            obs, info = self.env.reset(seed=seed)
            obs, info = self._random_opening(obs, info)
            obs, info, _, done = self._play_opponent_until_learner(obs, info)
            if not done:
                break
        info = {**info, "learner_color": self.learner_is}
        return obs, info

    def _random_opening(self, obs, info):
        """Play ``opening_plies`` uniformly-random legal plies (both sides) so the
        learner sees a diverse set of start positions and generalises."""
        n = self._rng.randint(0, self.opening_plies) if self.opening_plies else 0
        for _ in range(n):
            if self.env.state.is_terminal():
                break
            legal = np.nonzero(info["action_mask"])[0]
            obs, _, terminated, truncated, info = self.env.step(int(self._rng.choice(legal)))
            if terminated or truncated:
                break
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self.env.state.player == self.learner_is, "not the learner's turn"
        obs, reward, terminated, truncated, info = self.env.step(action)
        # reward is from the learner's perspective (learner just moved)
        if not (terminated or truncated):
            obs, info, opp_reward, opp_done = self._play_opponent_until_learner(obs, info)
            if opp_done:
                terminated = True
                reward = -opp_reward  # opponent's terminal reward, negated
                truncated = truncated or info.get("_truncated", False)
        info = {**info, "learner_color": self.learner_is}
        return obs, reward, terminated, truncated, info

    # -- internals --------------------------------------------------
    def _play_opponent_until_learner(self, obs, info):
        """Advance the base env through opponent plies. Returns
        ``(obs, info, opp_terminal_reward, opp_done)``."""
        opp_reward = 0.0
        done = self.env.state.is_terminal()
        truncated = False
        while not done and self.env.state.player != self.learner_is:
            move = self.opponent.select_move(self.env.state)
            opp_action = PASS_ACTION if move is None else move[0] * 8 + move[1]
            obs, r, terminated, truncated, info = self.env.step(opp_action)
            opp_reward = r  # from opponent's perspective (opponent just moved)
            done = terminated or truncated
        info = {**info, "_truncated": truncated}
        return obs, info, opp_reward, done

    def render(self) -> str:
        return self.env.render()

    @property
    def state(self):
        return self.env.state

"""Fixed-capacity uniform experience replay for masked DQN.

A transition is ``(obs, action, reward, next_obs, done, next_mask)``. ``next_mask``
(length-65 legal-action mask of ``next_obs``) is stored so the bootstrap target
can mask illegal actions in the next state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from othello_rl.environment.environment import NUM_ACTIONS, OBS_SHAPE


@dataclass
class Batch:
    obs: np.ndarray          # (B, 3, 8, 8) float32
    actions: np.ndarray      # (B,) int64
    rewards: np.ndarray      # (B,) float32
    next_obs: np.ndarray     # (B, 3, 8, 8) float32
    dones: np.ndarray        # (B,) float32
    next_masks: np.ndarray   # (B, 65) bool


class ReplayBuffer:
    def __init__(self, capacity: int, seed: Optional[int] = None):
        self.capacity = int(capacity)
        self._rng = np.random.default_rng(seed)
        self._obs = np.zeros((self.capacity, *OBS_SHAPE), dtype=np.float32)
        self._next_obs = np.zeros((self.capacity, *OBS_SHAPE), dtype=np.float32)
        self._actions = np.zeros(self.capacity, dtype=np.int64)
        self._rewards = np.zeros(self.capacity, dtype=np.float32)
        self._dones = np.zeros(self.capacity, dtype=np.float32)
        self._next_masks = np.zeros((self.capacity, NUM_ACTIONS), dtype=bool)
        self._size = 0
        self._pos = 0

    def __len__(self) -> int:
        return self._size

    def add(self, obs, action, reward, next_obs, done, next_mask) -> None:
        i = self._pos
        self._obs[i] = obs
        self._actions[i] = action
        self._rewards[i] = reward
        self._next_obs[i] = next_obs
        self._dones[i] = float(done)
        self._next_masks[i] = next_mask
        self._pos = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        if self._size < batch_size:
            raise ValueError(f"buffer has {self._size} < batch_size {batch_size}")
        idx = self._rng.integers(0, self._size, size=batch_size)
        return Batch(
            obs=self._obs[idx],
            actions=self._actions[idx],
            rewards=self._rewards[idx],
            next_obs=self._next_obs[idx],
            dones=self._dones[idx],
            next_masks=self._next_masks[idx],
        )

    def can_sample(self, batch_size: int) -> bool:
        return self._size >= batch_size

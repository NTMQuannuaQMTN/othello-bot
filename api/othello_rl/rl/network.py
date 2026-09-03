"""Small convolutional Q-network for 8x8 Othello.

Input : ``(B, 3, 8, 8)`` canonical observation (see
        :func:`othello_rl.environment.environment.encode_observation`).
Output: ``(B, 65)`` Q-values (0..63 = squares, 64 = pass).

A ``value`` head is included but unused by DQN; it is a hook for the later
AlphaZero-style upgrade and for optional auxiliary-loss experiments.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from othello_rl.environment.environment import NUM_ACTIONS

NEG_INF = -1e9  # finite sentinel so masked softmax/backprop stay well-defined


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=not norm)
        self.norm = nn.BatchNorm2d(out_ch) if norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.norm(self.conv(x)))


class SmallOthelloNet(nn.Module):
    def __init__(self, channels: int = 48, blocks: int = 3, hidden: int = 128,
                 num_actions: int = NUM_ACTIONS, with_value_head: bool = True,
                 norm: bool = False):
        super().__init__()
        self.num_actions = num_actions
        layers = [ConvBlock(3, channels, norm=norm)]
        for _ in range(blocks - 1):
            layers.append(ConvBlock(channels, channels, norm=norm))
        self.torso = nn.Sequential(*layers)
        self.flat = 8 * 8 * channels
        self.q_head = nn.Sequential(
            nn.Linear(self.flat, hidden), nn.ReLU(),
            nn.Linear(hidden, num_actions),
        )
        self.with_value_head = with_value_head
        if with_value_head:
            self.value_head = nn.Sequential(
                nn.Linear(self.flat, hidden), nn.ReLU(),
                nn.Linear(hidden, 1), nn.Tanh(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw Q-values ``(B, num_actions)`` (no masking applied)."""
        h = self.torso(x).flatten(1)
        return self.q_head(h)

    def forward_with_value(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        h = self.torso(x).flatten(1)
        q = self.q_head(h)
        v = self.value_head(h).squeeze(-1) if self.with_value_head else None
        return q, v


def masked_q(q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Set Q-values of illegal actions to a large negative sentinel.

    ``mask`` is a boolean/0-1 tensor broadcastable to ``q`` where True == legal.
    """
    mask_bool = mask.bool()
    return torch.where(mask_bool, q, torch.full_like(q, NEG_INF))


def greedy_action(q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Argmax over legal actions. ``q``/``mask`` shape ``(B, num_actions)``."""
    return masked_q(q, mask).argmax(dim=-1)

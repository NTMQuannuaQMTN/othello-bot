"""AlphaZero-style policy(+value) network: one shared conv torso feeding a 65-way
policy head (move logits) and a scalar value head (tanh, game outcome).

Used for **supervised imitation pretraining** on historical games
(``rl/supervised.py``) and, later, as the net for AZ-style MCTS self-play
(``docs/alphazero-plan.md``). Distinct from the DQN ``SmallOthelloNet`` — the two
model kinds coexist behind ``rl/checkpoint.py::load_agent``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from othello_rl.environment.environment import NUM_ACTIONS

from .network import ConvBlock


@dataclass
class PolicyValueConfig:
    channels: int = 48
    blocks: int = 4
    hidden: int = 128
    norm: bool = True

    def build(self) -> "PolicyValueNet":
        return PolicyValueNet(channels=self.channels, blocks=self.blocks,
                              hidden=self.hidden, norm=self.norm)


class PolicyValueNet(nn.Module):
    def __init__(self, channels: int = 48, blocks: int = 4, hidden: int = 128,
                 norm: bool = True, num_actions: int = NUM_ACTIONS):
        super().__init__()
        self.num_actions = num_actions
        layers = [ConvBlock(3, channels, norm=norm)]
        for _ in range(blocks - 1):
            layers.append(ConvBlock(channels, channels, norm=norm))
        self.torso = nn.Sequential(*layers)
        flat = 8 * 8 * channels
        self.policy_head = nn.Sequential(
            nn.Linear(flat, hidden), nn.ReLU(), nn.Linear(hidden, num_actions))
        self.value_head = nn.Sequential(
            nn.Linear(flat, hidden), nn.ReLU(), nn.Linear(hidden, 1), nn.Tanh())

    def forward(self, x: torch.Tensor):
        """``(policy_logits (B, 65), value (B,))``."""
        h = self.torso(x).flatten(1)
        return self.policy_head(h), self.value_head(h).squeeze(-1)

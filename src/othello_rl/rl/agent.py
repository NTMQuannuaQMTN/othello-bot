"""DQN agent: wraps a Q-network for both training-time action selection
(masked epsilon-greedy) and evaluation / tournament play (masked greedy).

Also implements the baseline :class:`~othello_rl.agents.base.Agent` interface so a
checkpoint can be dropped straight into the evaluation framework.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from othello_rl.agents.base import Agent, Move
from othello_rl.environment.board import BOARD_SIZE, PASS_ACTION, Board
from othello_rl.environment.environment import encode_observation, legal_action_mask
from .network import SmallOthelloNet, greedy_action, masked_q


@dataclass
class NetworkConfig:
    channels: int = 48
    blocks: int = 3
    hidden: int = 128
    with_value_head: bool = True
    norm: bool = False

    def build(self) -> SmallOthelloNet:
        return SmallOthelloNet(
            channels=self.channels, blocks=self.blocks, hidden=self.hidden,
            with_value_head=self.with_value_head, norm=self.norm,
        )


@dataclass
class AgentMeta:
    train_steps: int = 0
    env_steps: int = 0
    episodes: int = 0
    extra: Dict = field(default_factory=dict)


class DQNAgent(Agent):
    name = "dqn"

    def __init__(self, net_config: Optional[NetworkConfig] = None,
                 device: str = "cpu", seed: Optional[int] = None,
                 net: Optional[SmallOthelloNet] = None):
        self.net_config = net_config or NetworkConfig()
        self.device = torch.device(device)
        self.net = (net or self.net_config.build()).to(self.device)
        self.meta = AgentMeta()
        self._rng = random.Random(seed)

    # -- training-time action selection --------------------------------
    @torch.no_grad()
    def act(self, obs: np.ndarray, mask: np.ndarray, epsilon: float = 0.0) -> int:
        legal = np.nonzero(mask)[0]
        if len(legal) == 0:  # terminal state slipped through
            return PASS_ACTION
        if epsilon > 0.0 and self._rng.random() < epsilon:
            return int(self._rng.choice(legal.tolist()))
        return self.greedy_act(obs, mask)

    @torch.no_grad()
    def greedy_act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        self.net.eval()
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        m = torch.as_tensor(np.asarray(mask), dtype=torch.bool, device=self.device).unsqueeze(0)
        q = self.net(o)
        return int(greedy_action(q, m).item())

    @torch.no_grad()
    def q_values(self, obs: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        self.net.eval()
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.net(o)
        if mask is not None:
            m = torch.as_tensor(np.asarray(mask), dtype=torch.bool, device=self.device).unsqueeze(0)
            q = masked_q(q, m)
        return q.squeeze(0).cpu().numpy()

    # -- baseline Agent interface (greedy, deterministic) --------------
    def select_move(self, state: Board) -> Move:
        moves = state.legal_moves()
        if not moves:
            return None
        obs = encode_observation(state)
        mask = legal_action_mask(state)
        action = self.greedy_act(obs, mask)
        if action == PASS_ACTION:  # shouldn't happen when moves exist
            return moves[0]
        return divmod(action, BOARD_SIZE)

    # -- checkpointing ------------------------------------------------
    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "format": 1,
            "net_config": asdict(self.net_config),
            "state_dict": self.net.state_dict(),
            "meta": asdict(self.meta),
        }, path)

    def load(self, path, strict: bool = True) -> "DQNAgent":
        ckpt = torch.load(Path(path), map_location=self.device, weights_only=False)
        self.net_config = NetworkConfig(**ckpt["net_config"])
        self.net = self.net_config.build().to(self.device)
        self.net.load_state_dict(ckpt["state_dict"], strict=strict)
        m = ckpt.get("meta", {})
        self.meta = AgentMeta(train_steps=m.get("train_steps", 0),
                              env_steps=m.get("env_steps", 0),
                              episodes=m.get("episodes", 0),
                              extra=m.get("extra", {}))
        return self

    @classmethod
    def from_checkpoint(cls, path, device: str = "cpu", seed: Optional[int] = None) -> "DQNAgent":
        agent = cls(device=device, seed=seed)
        agent.load(path)
        agent.name = f"dqn@{Path(path).stem}"
        return agent

    def clone_network(self) -> SmallOthelloNet:
        """A detached copy of the current network (for target nets / opponent pool)."""
        twin = self.net_config.build().to(self.device)
        twin.load_state_dict(self.net.state_dict())
        twin.eval()
        for p in twin.parameters():
            p.requires_grad_(False)
        return twin

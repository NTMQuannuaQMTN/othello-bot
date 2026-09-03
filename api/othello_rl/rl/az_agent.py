"""A move-selecting agent backed by a :class:`PolicyValueNet` — masked-argmax over
the policy head. Implements the project ``Agent`` interface, so it plugs straight
into ``evaluation/tournament.py`` / ``scripts/eval_bot.py`` / ``promote_model.py``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from othello_rl.agents.base import Agent, Move
from othello_rl.environment.board import BOARD_SIZE, Board
from othello_rl.environment.environment import encode_observation, legal_action_mask

from .az_network import PolicyValueConfig, PolicyValueNet


class PolicyValueAgent(Agent):
    name = "policy-value"

    def __init__(self, net: PolicyValueNet, device: str = "cpu",
                 config: Optional[PolicyValueConfig] = None):
        self.net = net.to(device).eval()
        self.device = torch.device(device)
        self.config = config

    @torch.no_grad()
    def _eval(self, board: Board):
        obs = torch.as_tensor(encode_observation(board), dtype=torch.float32,
                              device=self.device).unsqueeze(0)
        logits, value = self.net(obs)
        return logits.squeeze(0).cpu().numpy(), float(value.item())

    def select_move(self, state: Board) -> Move:
        moves = state.legal_moves()
        if not moves:
            return None
        logits, _ = self._eval(state)
        mask = np.asarray(legal_action_mask(state), dtype=bool)
        masked = np.where(mask, logits, -np.inf)
        a = int(np.argmax(masked))
        if a >= BOARD_SIZE * BOARD_SIZE:      # 64 = pass; shouldn't happen with moves
            return moves[0]
        return divmod(a, BOARD_SIZE)

    def move_probabilities(self, board: Board) -> dict:
        logits, _ = self._eval(board)
        mask = np.asarray(legal_action_mask(board), dtype=bool)
        z = np.where(mask, logits, -np.inf)
        z = z - np.max(z[mask])
        p = np.where(mask, np.exp(z), 0.0)
        p = p / p.sum()
        return {int(i): float(p[i]) for i in np.nonzero(mask)[0]}

    def value(self, board: Board) -> float:
        return self._eval(board)[1]

    # -- construction --------------------------------------------------
    @classmethod
    def from_state(cls, pv_config: dict, state_dict, device: str = "cpu") -> "PolicyValueAgent":
        cfg = PolicyValueConfig(**pv_config)
        net = cfg.build()
        net.load_state_dict(state_dict)
        return cls(net, device=device, config=cfg)

    @classmethod
    def load(cls, path, device: str = "cpu") -> "PolicyValueAgent":
        raw = torch.load(Path(path), map_location=device, weights_only=False)
        return cls.from_state(raw["pv_config"], raw["state_dict"], device=device)

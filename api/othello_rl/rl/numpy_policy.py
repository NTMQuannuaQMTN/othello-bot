"""Torch-free inference for :class:`~othello_rl.rl.network.SmallOthelloNet`.

The production net is tiny (two 3x3 conv layers + two linear heads, no
BatchNorm), so its forward pass is a few lines of numpy.  This lets the web app
serve moves/grades/eval **without importing PyTorch** — which is what makes the
Vercel deploy (250 MB function limit) possible.

Weights come from an ``.npz`` written by ``scripts/export_policy.py`` (run once,
with the full install).  This module never imports torch.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from othello_rl.agents.base import Agent, Move
from othello_rl.environment.board import BOARD_SIZE, PASS_ACTION, Board
from othello_rl.environment.environment import encode_observation, legal_action_mask

_NEG_INF = -1e9


def _conv3x3(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``x`` (Cin, 8, 8), ``w`` (Cout, Cin, 3, 3), ``b`` (Cout,) -> (Cout, 8, 8),
    same-padding, matching ``nn.Conv2d(.., 3, padding=1)``."""
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    out = np.zeros((w.shape[0], 8, 8), dtype=np.float32)
    for i in range(3):
        for j in range(3):
            out += np.einsum("oc,chw->ohw", w[:, :, i, j], xp[:, i:i + 8, j:j + 8],
                             optimize=True)
    return out + b[:, None, None]


class _NetMeta:
    """Enough of :class:`AgentMeta` for :meth:`OthelloBot.describe`."""

    def __init__(self, train_steps=0, env_steps=0, episodes=0, extra=None):
        self.train_steps = int(train_steps)
        self.env_steps = int(env_steps)
        self.episodes = int(episodes)
        self.extra = dict(extra or {})


class NumpyPolicy(Agent):
    """Drop-in for the read-only bits of :class:`DQNAgent` (``q_values`` /
    ``greedy_act`` / ``select_move``).  Fine-tuning is not supported."""

    name = "dqn"

    def __init__(self, weights: dict, net_config: dict, meta: Optional[dict] = None):
        self.net_config = dict(net_config)
        self.device = "cpu"
        m = meta or {}
        self.meta = _NetMeta(m.get("train_steps", 0), m.get("env_steps", 0),
                             m.get("episodes", 0), m.get("extra", {}))
        f32 = lambda a: np.ascontiguousarray(a, dtype=np.float32)  # noqa: E731
        # conv stack: torso.0, torso.1, ...
        self._convs = []
        i = 0
        while f"torso.{i}.conv.weight" in weights:
            self._convs.append((f32(weights[f"torso.{i}.conv.weight"]),
                                f32(weights[f"torso.{i}.conv.bias"])))
            i += 1
        # linear weights stored transposed + contiguous (y = x @ Wt + b)
        self._q = ((f32(weights["q_head.0.weight"].T), f32(weights["q_head.0.bias"])),
                   (f32(weights["q_head.2.weight"].T), f32(weights["q_head.2.bias"])))
        self.with_value_head = "value_head.0.weight" in weights
        self.param_count = int(sum(w.size + b.size for w, b in self._convs)
                               + sum(w.size + b.size for w, b in self._q)
                               + sum(int(v.size) for k, v in weights.items()
                                     if k.startswith("value_head.")))

    # -- loading ------------------------------------------------------
    @classmethod
    def load(cls, path) -> "NumpyPolicy":
        """``path`` is a directory or an ``.npz`` written by export_policy.py."""
        p = Path(path)
        npz = p / "policy.npz" if p.is_dir() else p
        data = np.load(npz, allow_pickle=False)
        cfg_raw = data["net_config_json"].item() if "net_config_json" in data else None
        import json
        cfg = json.loads(cfg_raw) if cfg_raw else {"channels": 32, "blocks": 2, "hidden": 96}
        meta = json.loads(data["meta_json"].item()) if "meta_json" in data else {}
        weights = {k: data[k] for k in data.files if k.startswith(("torso.", "q_head.", "value_head."))}
        return cls(weights, cfg, meta)

    # -- inference --------------------------------------------------
    def _forward_q(self, obs: np.ndarray) -> np.ndarray:
        # macOS Accelerate raises spurious fp warnings on small float32 matmuls;
        # the result is bit-for-bit fine (export_policy.py checks it vs torch).
        with np.errstate(all="ignore"):
            h = obs.astype(np.float32)
            for w, b in self._convs:
                h = np.maximum(_conv3x3(h, w, b), 0.0)
            h = np.ascontiguousarray(h.reshape(-1))        # (C*8*8,), C-major — matches torch
            (wt0, b0), (wt1, b1) = self._q
            h = np.maximum(h @ wt0 + b0, 0.0)
            return (h @ wt1 + b1).astype(np.float32)

    def q_values(self, obs: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        q = self._forward_q(np.asarray(obs))
        if mask is not None:
            q = np.where(np.asarray(mask, dtype=bool), q, _NEG_INF)
        return q

    def greedy_act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        legal = np.nonzero(np.asarray(mask))[0]
        if len(legal) == 0:
            return PASS_ACTION
        return int(np.argmax(self.q_values(obs, mask)))

    def act(self, obs: np.ndarray, mask: np.ndarray, epsilon: float = 0.0) -> int:
        return self.greedy_act(obs, mask)

    # -- baseline Agent interface --------------------------------
    def select_move(self, state: Board) -> Move:
        moves = state.legal_moves()
        if not moves:
            return None
        a = self.greedy_act(encode_observation(state), legal_action_mask(state))
        return moves[0] if a == PASS_ACTION else divmod(a, BOARD_SIZE)

"""Supervised **imitation / behaviour-cloning** pretraining of a
:class:`PolicyValueNet` on historical games. This is NOT reinforcement learning —
the target is a fixed (position -> move) mapping plus the game outcome.

The dataset is a ``data/processed/training_data/<version>/`` directory of
``{train,val,test}.npz`` from ``scripts/build_dataset.py``. Splitting is at the
game level there, so the val split shares no game with train.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .az_network import PolicyValueConfig, PolicyValueNet


@dataclass
class SupervisedConfig:
    az_network: dict = field(default_factory=lambda: {"channels": 48, "blocks": 4,
                                                      "hidden": 128, "norm": True})
    epochs: int = 20
    lr: float = 1e-3
    batch_size: int = 512
    weight_decay: float = 1e-4
    value_loss_weight: float = 0.5
    seed: int = 20260901

    def net_config(self) -> PolicyValueConfig:
        return PolicyValueConfig(**self.az_network)


def load_split(npz_path) -> Dict[str, np.ndarray]:
    d = np.load(Path(npz_path), allow_pickle=True)
    return {k: d[k] for k in ("obs", "policy", "value", "weight")}


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_policy_loss: float
    train_value_loss: float
    val_loss: float
    val_policy_loss: float
    val_value_loss: float
    move_accuracy: float           # top-1 policy accuracy on the val split
    seconds: float


class SupervisedTrainer:
    def __init__(self, cfg: SupervisedConfig, net: Optional[PolicyValueNet] = None,
                 device: str = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.net = (net or cfg.net_config().build()).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr,
                                    weight_decay=cfg.weight_decay)
        self.epoch = 0
        self._rng = random.Random(cfg.seed)
        torch.manual_seed(cfg.seed)

    # -- loss ---------------------------------------------------------
    def _losses(self, obs, policy, value, weight):
        logits, v = self.net(obs)
        w = weight / weight.sum().clamp(min=1e-6)
        pol = (F.cross_entropy(logits, policy, reduction="none") * w).sum()
        val = (F.mse_loss(v, value, reduction="none") * w).sum()
        return pol + self.cfg.value_loss_weight * val, pol, val, logits

    # -- data -------------------------------------------------------
    def _tensors(self, split):
        if not isinstance(split, dict):
            split = load_split(split)
        return (torch.as_tensor(split["obs"], dtype=torch.float32, device=self.device),
                torch.as_tensor(split["policy"].astype(np.int64), device=self.device),
                torch.as_tensor(split["value"], dtype=torch.float32, device=self.device),
                torch.as_tensor(split["weight"], dtype=torch.float32, device=self.device))

    def _batches(self, n: int, shuffle: bool):
        idx = list(range(n))
        if shuffle:
            self._rng.shuffle(idx)
        for i in range(0, n, self.cfg.batch_size):
            yield torch.as_tensor(idx[i:i + self.cfg.batch_size], device=self.device)

    # -- fit -------------------------------------------------------
    def fit(self, train_npz, val_npz, *, on_epoch=None) -> "list[EpochMetrics]":
        tr = self._tensors(train_npz)
        va = self._tensors(val_npz)
        history = []
        target = self.epoch + self.cfg.epochs
        while self.epoch < target:
            t0 = time.time()
            self.net.train()
            tl = tp = tv = 0.0
            nb = 0
            for b in self._batches(tr[0].shape[0], shuffle=True):
                loss, pol, val, _ = self._losses(tr[0][b], tr[1][b], tr[2][b], tr[3][b])
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                self.opt.step()
                tl += loss.item(); tp += pol.item(); tv += val.item(); nb += 1
            self.epoch += 1
            vm = self.evaluate(va)
            m = EpochMetrics(
                epoch=self.epoch, train_loss=tl / nb, train_policy_loss=tp / nb,
                train_value_loss=tv / nb, seconds=round(time.time() - t0, 2), **vm)
            history.append(m)
            if on_epoch:
                on_epoch(m)
        return history

    @torch.no_grad()
    def evaluate(self, val) -> dict:
        if isinstance(val, tuple):
            obs, policy, value, weight = val
        elif isinstance(val, dict):
            obs, policy, value, weight = self._tensors(val)
        else:  # a path
            obs, policy, value, weight = self._tensors(load_split(val))
        self.net.eval()
        loss = pol = valn = 0.0
        correct = total = 0
        nb = 0
        for b in self._batches(obs.shape[0], shuffle=False):
            l, p, vv, logits = self._losses(obs[b], policy[b], value[b], weight[b])
            loss += float(l); pol += float(p); valn += float(vv); nb += 1
            correct += int((logits.argmax(1) == policy[b]).sum())
            total += int(b.shape[0])
        return {"val_loss": loss / nb, "val_policy_loss": pol / nb,
                "val_value_loss": valn / nb,
                "move_accuracy": correct / total if total else 0.0}

    def rng_state(self) -> dict:
        return {"python": self._rng.getstate(), "torch": torch.get_rng_state()}

    # -- resume -----------------------------------------------------
    @classmethod
    def resume(cls, cfg: SupervisedConfig, checkpoint_path, device: str = "cpu"
               ) -> "SupervisedTrainer":
        from .checkpoint import load_checkpoint
        ck = load_checkpoint(checkpoint_path)
        if ck.net_kind != "policy_value":
            raise ValueError(f"{checkpoint_path}: not a policy_value checkpoint")
        net = PolicyValueConfig(**ck.pv_config).build()
        net.load_state_dict(ck.state_dict)
        trainer = cls(cfg, net=net, device=device)
        if ck.optimizer_state is not None:
            trainer.opt.load_state_dict(ck.optimizer_state)
        trainer.epoch = int(ck.meta.extra.get("epoch", ck.train_step))
        rs = ck.rng_state or {}
        if rs.get("python") is not None:
            trainer._rng.setstate(_as_tuple(rs["python"]))
        if rs.get("torch") is not None:
            t = rs["torch"]
            torch.set_rng_state(t if isinstance(t, torch.Tensor)
                                else torch.as_tensor(t, dtype=torch.uint8))
        return trainer


def _as_tuple(obj):
    if isinstance(obj, list):
        return tuple(_as_tuple(x) for x in obj)
    return obj

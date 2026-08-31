"""Masked Double-DQN trainer.

Expects a *single-agent* env (e.g. :class:`~othello_rl.rl.opponents.FixedOpponentEnv`
or a self-play wrapper) exposing ``reset() -> (obs, info)`` and
``step(action) -> (obs, reward, terminated, truncated, info)`` with
``info["action_mask"]`` a length-65 boolean mask.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .agent import DQNAgent
from .network import masked_q
from .replay_buffer import ReplayBuffer


@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 256
    buffer_capacity: int = 100_000
    warmup_steps: int = 2_000
    train_freq: int = 1
    target_sync: int = 1_000
    grad_clip: float = 10.0
    double_dqn: bool = True
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 30_000

    def epsilon(self, step: int) -> float:
        if step >= self.epsilon_decay_steps:
            return self.epsilon_end
        frac = step / max(1, self.epsilon_decay_steps)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)


@dataclass
class TrainMetrics:
    history: List[Dict[str, float]] = field(default_factory=list)

    def log(self, **row: float) -> None:
        self.history.append(row)


class DQNTrainer:
    def __init__(self, env, agent: DQNAgent, config: Optional[DQNConfig] = None,
                 seed: Optional[int] = None, resume_state=None):
        self.env = env
        self.agent = agent
        self.cfg = config or DQNConfig()
        self.device = agent.device
        self.target = agent.clone_network()
        self.opt = torch.optim.Adam(agent.net.parameters(), lr=self.cfg.lr)
        self.buffer = ReplayBuffer(self.cfg.buffer_capacity, seed=seed)
        self.metrics = TrainMetrics()

        self.env_steps = 0
        self.train_steps = 0
        self.episodes = 0
        self._ep_return = 0.0
        self._returns = deque(maxlen=100)
        self._obs, self._info = self.env.reset(seed=seed)
        if resume_state is not None:
            self.load_resume_state(resume_state)

    # -- resume (weights are restored on the agent by the caller) -------
    def load_resume_state(self, rs) -> None:
        """Adopt counters + optimizer + RNG from a
        :class:`othello_rl.rl.checkpoint.ResumeState`. The replay buffer is not
        restored; it re-warms from fresh rollouts."""
        from .checkpoint import restore_rng_state
        self.env_steps = int(rs.env_steps)
        self.train_steps = int(rs.train_steps)
        self.episodes = int(rs.episodes)
        if getattr(rs, "optimizer_state", None) is not None:
            self.opt.load_state_dict(rs.optimizer_state)
        self.target.load_state_dict(self.agent.net.state_dict())
        restore_rng_state(getattr(rs, "rng_state", None))

    def training_state(self) -> Dict[str, object]:
        return {"env_steps": self.env_steps, "train_steps": self.train_steps,
                "episodes": self.episodes, "optimizer_state": self.opt.state_dict()}

    # -- rollout ----------------------------------------------------
    def collect_step(self) -> None:
        mask = np.asarray(self._info["action_mask"])
        eps = self.cfg.epsilon(self.env_steps)
        action = self.agent.act(self._obs, mask, epsilon=eps)
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)
        next_mask = np.asarray(info["action_mask"])
        self.buffer.add(self._obs, action, reward, next_obs, done, next_mask)

        self.env_steps += 1
        self._ep_return += reward
        if done:
            self.episodes += 1
            self._returns.append(self._ep_return)
            self._ep_return = 0.0
            self._obs, self._info = self.env.reset()
        else:
            self._obs, self._info = next_obs, info

    # -- learning --------------------------------------------------
    def train_step(self) -> Optional[float]:
        if not self.buffer.can_sample(self.cfg.batch_size):
            return None
        b = self.buffer.sample(self.cfg.batch_size)
        dev = self.device
        obs = torch.as_tensor(b.obs, device=dev)
        actions = torch.as_tensor(b.actions, device=dev)
        rewards = torch.as_tensor(b.rewards, device=dev)
        next_obs = torch.as_tensor(b.next_obs, device=dev)
        dones = torch.as_tensor(b.dones, device=dev)
        next_masks = torch.as_tensor(b.next_masks, device=dev)

        self.agent.net.train()
        q = self.agent.net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_target = self.target(next_obs)
            if self.cfg.double_dqn:
                next_q_online = masked_q(self.agent.net(next_obs), next_masks)
                next_actions = next_q_online.argmax(dim=1, keepdim=True)
                next_v = masked_q(next_q_target, next_masks).gather(1, next_actions).squeeze(1)
            else:
                next_v = masked_q(next_q_target, next_masks).max(dim=1).values
            # A terminal next_obs has an all-False mask -> next_v is the NEG_INF
            # sentinel, but every such transition also has done==1, so the
            # (1 - dones) factor zeroes it out. Clamp as belt-and-braces.
            next_v = torch.where(dones.bool(), torch.zeros_like(next_v), next_v)
            target = rewards + self.cfg.gamma * (1.0 - dones) * next_v

        loss = F.smooth_l1_loss(q, target)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        if self.cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.agent.net.parameters(), self.cfg.grad_clip)
        self.opt.step()

        self.train_steps += 1
        if self.train_steps % self.cfg.target_sync == 0:
            self.target.load_state_dict(self.agent.net.state_dict())
        return float(loss.item())

    # -- driver ---------------------------------------------------
    def learn(self, total_env_steps: int,
              eval_fn: Optional[Callable[["DQNTrainer"], Dict[str, float]]] = None,
              eval_every: int = 5_000, log_every: int = 1_000,
              progress: "bool | str" = False, pbar=None) -> TrainMetrics:
        """Run ``total_env_steps`` of collect + learn.

        ``pbar`` : an external progress bar (from
        :func:`othello_rl.utils.progress.make_progress`) to advance; it is *not*
        closed here. If ``pbar`` is None and ``progress`` is truthy a local bar is
        created and closed.
        """
        own_bar = pbar is None
        if own_bar:
            from othello_rl.utils.progress import make_progress
            pbar = make_progress(total_env_steps, enabled=progress or False)

        start = time.time()
        last_loss = float("nan")
        target_steps = self.env_steps + total_env_steps
        while self.env_steps < target_steps:
            self.collect_step()
            if (self.env_steps > self.cfg.warmup_steps
                    and self.env_steps % self.cfg.train_freq == 0):
                l = self.train_step()
                if l is not None:
                    last_loss = l
            pbar.update(1)

            if self.env_steps % log_every == 0:
                sps = self.env_steps / max(1e-6, time.time() - start)
                mret = float(np.mean(self._returns)) if self._returns else 0.0
                row = {
                    "env_steps": self.env_steps,
                    "train_steps": self.train_steps,
                    "episodes": self.episodes,
                    "epsilon": self.cfg.epsilon(self.env_steps),
                    "loss": last_loss,
                    "mean_return_100": mret,
                    "sps": sps,
                }
                self.metrics.log(**row)
                pbar.set_postfix({"eps": round(row["epsilon"], 3),
                                  "loss": round(last_loss, 4) if last_loss == last_loss else "-",
                                  "ret100": round(mret, 3)})

            if eval_fn is not None and self.env_steps % eval_every == 0:
                self._sync_agent_meta()
                eval_row = eval_fn(self)
                eval_row = {"env_steps": self.env_steps, **eval_row}
                self.metrics.log(**eval_row)

        if own_bar:
            pbar.close()
        self._sync_agent_meta()
        return self.metrics

    def _sync_agent_meta(self) -> None:
        self.agent.meta.train_steps = self.train_steps
        self.agent.meta.env_steps = self.env_steps
        self.agent.meta.episodes = self.episodes

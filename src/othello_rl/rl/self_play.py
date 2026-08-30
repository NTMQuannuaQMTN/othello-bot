"""Self-play with a mixed opponent pool.

Naive "newest vs newest forever" self-play is unstable and forgets how to beat
weaker/older strategies. Instead we sample each episode's opponent from a pool
with three configurable categories:

- ``baseline``   : the fixed non-RL agents (random / greedy / heuristic / ...)
- ``historical`` : periodic frozen snapshots of the learner from early -> late
- ``recent``     : the last few frozen snapshots of the learner

A starting distribution of ``{baseline: 0.2, historical: 0.3, recent: 0.5}`` is
provided but every weight is configurable.
"""
from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from othello_rl.agents import Agent, make_agent
from .agent import DQNAgent, NetworkConfig

DEFAULT_DISTRIBUTION = {"baseline": 0.2, "historical": 0.3, "recent": 0.5}


@dataclass
class OpponentPool:
    baseline_specs: Sequence[str] = ("random", "greedy", "heuristic")
    distribution: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_DISTRIBUTION))
    recent_capacity: int = 5
    historical_every: int = 3          # every Nth snapshot is also kept as historical
    seed: Optional[int] = None

    def __post_init__(self):
        self._rng = random.Random(self.seed)
        self._recent: List[DQNAgent] = []
        self._historical: List[DQNAgent] = []
        self._snapshot_count = 0

    # -- population ---------------------------------------------------
    def add_snapshot(self, agent: DQNAgent, tag: Optional[str] = None) -> DQNAgent:
        """Freeze the current learner and add it to the pool."""
        frozen = _freeze(agent, tag or f"snap{self._snapshot_count}")
        self._recent.append(frozen)
        if len(self._recent) > self.recent_capacity:
            self._recent.pop(0)
        if self._snapshot_count % self.historical_every == 0:
            self._historical.append(frozen)
        self._snapshot_count += 1
        return frozen

    # -- sampling ---------------------------------------------------
    def _available_categories(self) -> Dict[str, float]:
        avail = {}
        if self.baseline_specs and self.distribution.get("baseline", 0) > 0:
            avail["baseline"] = self.distribution["baseline"]
        if self._historical and self.distribution.get("historical", 0) > 0:
            avail["historical"] = self.distribution["historical"]
        if self._recent and self.distribution.get("recent", 0) > 0:
            avail["recent"] = self.distribution["recent"]
        return avail

    def sample_with_category(self, rng: Optional[random.Random] = None):
        r = rng or self._rng
        avail = self._available_categories()
        if not avail:  # pool has no snapshots yet -> fall back to baselines
            spec = r.choice(list(self.baseline_specs))
            return make_agent(spec, seed=r.randrange(2 ** 31)), "baseline"
        cats, weights = zip(*avail.items())
        category = r.choices(cats, weights=weights, k=1)[0]
        if category == "baseline":
            spec = r.choice(list(self.baseline_specs))
            return make_agent(spec, seed=r.randrange(2 ** 31)), "baseline"
        pool = self._historical if category == "historical" else self._recent
        return r.choice(pool), category

    def sample(self, rng: Optional[random.Random] = None) -> Agent:
        return self.sample_with_category(rng)[0]

    # -- introspection / persistence ------------------------------
    @property
    def num_recent(self) -> int:
        return len(self._recent)

    @property
    def num_historical(self) -> int:
        return len(self._historical)

    def category_counts(self, n: int = 2000, seed: int = 0) -> Dict[str, int]:
        """Empirical category frequencies over ``n`` samples (for tests/plots)."""
        rng = random.Random(seed)
        counts: Dict[str, int] = {"baseline": 0, "historical": 0, "recent": 0}
        for _ in range(n):
            _, category = self.sample_with_category(rng)
            counts[category] += 1
        return counts

    # -- persistence (for resuming a self-play run) ----------------
    def save(self, path) -> None:
        """Write the pool's frozen snapshots + counters so a run can resume."""
        nc = self._recent[0].net_config if self._recent else NetworkConfig()
        torch.save({
            "snapshot_count": self._snapshot_count,
            "net_config": asdict(nc),
            "recent": [(a.name, a.net.state_dict()) for a in self._recent],
            "historical": [(a.name, a.net.state_dict()) for a in self._historical],
        }, Path(path))

    def load(self, path, device: str = "cpu") -> "OpponentPool":
        ckpt = torch.load(Path(path), map_location=device, weights_only=False)
        nc = NetworkConfig(**ckpt["net_config"])

        def rebuild(name, state_dict) -> DQNAgent:
            ag = DQNAgent(nc, device=device)
            ag.net.load_state_dict(state_dict)
            ag.net.eval()
            for p in ag.net.parameters():
                p.requires_grad_(False)
            ag.name = name
            return ag

        self._recent = [rebuild(n, sd) for n, sd in ckpt["recent"]]
        self._historical = [rebuild(n, sd) for n, sd in ckpt["historical"]]
        self._snapshot_count = int(ckpt["snapshot_count"])
        return self


@dataclass
class SelfPlayConfig:
    total_env_steps: int = 200_000
    snapshot_every: int = 20_000
    eval_every: int = 20_000
    eval_games: int = 80
    eval_seed: int = 777
    checkpoint_every: int = 40_000
    learner_color: str = "random"
    opening_plies: int = 4
    pool: "OpponentPool" = field(default_factory=OpponentPool)
    dqn: "object" = None  # DQNConfig; imported lazily to avoid a cycle


def run_self_play(agent: DQNAgent, cfg: SelfPlayConfig, run_dir, seed: int = 0,
                  progress: "bool | str" = "auto", resume_pool: Optional[str] = None):
    """Train ``agent`` by self-play against ``cfg.pool``, snapshotting the learner
    into the pool periodically and evaluating against baselines + historical
    snapshots (anti-forgetting).

    ``resume_pool`` : path to a ``pool.pt`` from a previous run — restores the
    opponent pool so a warm-started run continues rather than restarting the
    historical/recent ladder.
    """
    from othello_rl.evaluation.harness import evaluate_agent, flatten_eval
    from othello_rl.utils.logging import MetricLogger
    from othello_rl.utils.progress import make_progress
    from .opponents import FixedOpponentEnv
    from .trainer import DQNConfig, DQNTrainer

    run_dir = Path(run_dir)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger = MetricLogger(run_dir / "metrics.jsonl")

    dqn_cfg = cfg.dqn or DQNConfig()
    pool = cfg.pool
    if resume_pool:
        pool.load(resume_pool, device=str(agent.device))
        logger.log(phase="resume", pool=str(resume_pool),
                   recent=pool.num_recent, historical=pool.num_historical)
    else:
        # seed the pool with the starting (near-random) agent so early self-play
        # has someone to play before the first scheduled snapshot
        pool.add_snapshot(agent, tag="init")

    env = FixedOpponentEnv(pool, learner_color=cfg.learner_color, seed=seed,
                           opening_plies=cfg.opening_plies)
    trainer = DQNTrainer(env, agent, dqn_cfg, seed=seed)

    baseline_opponents = {s: s for s in pool.baseline_specs}

    bar = make_progress(cfg.total_env_steps, enabled=progress, desc="self-play")

    def do_eval(where: str):
        agent.net.eval()
        res = evaluate_agent(agent, baseline_opponents, num_games=cfg.eval_games, seed=cfg.eval_seed)
        row = {"phase": "eval", "where": where, "env_steps": trainer.env_steps,
               "train_steps": trainer.train_steps, **flatten_eval(res)}
        # anti-forgetting: vs the oldest and a mid historical snapshot
        hist = pool._historical
        if hist:
            targets = {f"hist0": hist[0]}
            if len(hist) > 2:
                targets["hist_mid"] = hist[len(hist) // 2]
            hres = evaluate_agent(agent, targets, num_games=cfg.eval_games // 2, seed=cfg.eval_seed + 1)
            row.update(flatten_eval(hres))
        logger.log(**row)
        parts = " ".join(f"{k[len('winrate_vs_'):]}={v:.2f}"
                         for k, v in row.items() if k.startswith("winrate_vs_"))
        bar.write(f"[self-play] eval ({where}) @ {trainer.env_steps} steps:  {parts}")
        return row

    do_eval("start")
    agent.save(ckpt_dir / "start.pt")

    last_snap = last_eval = last_ckpt = 0
    while trainer.env_steps < cfg.total_env_steps:
        chunk = min(2000, cfg.total_env_steps - trainer.env_steps)
        trainer.learn(total_env_steps=chunk, log_every=2000, pbar=bar)
        s = trainer.env_steps
        if s - last_snap >= cfg.snapshot_every:
            last_snap = s
            trainer._sync_agent_meta()
            pool.add_snapshot(agent, tag=f"step{s}")
            bar.write(f"[self-play] snapshot @ {s} steps  "
                      f"(recent={pool.num_recent}, historical={pool.num_historical})")
            logger.log(phase="snapshot", env_steps=s, recent=pool.num_recent,
                       historical=pool.num_historical)
        if s - last_eval >= cfg.eval_every:
            last_eval = s
            do_eval("periodic")
        if s - last_ckpt >= cfg.checkpoint_every:
            last_ckpt = s
            trainer._sync_agent_meta()
            agent.save(ckpt_dir / f"step{s}.pt")
            pool.save(ckpt_dir / "pool.pt")

    trainer._sync_agent_meta()
    do_eval("end")
    bar.close()
    agent.save(ckpt_dir / "final.pt")
    pool.save(ckpt_dir / "pool.pt")
    logger.log(phase="done", env_steps=trainer.env_steps)
    return logger


def _freeze(agent: DQNAgent, tag: str) -> DQNAgent:
    frozen = DQNAgent(agent.net_config, device=str(agent.device))
    frozen.net = agent.clone_network()
    frozen.net.eval()
    frozen.meta = copy.deepcopy(agent.meta)
    frozen.name = f"dqn:{tag}"
    return frozen

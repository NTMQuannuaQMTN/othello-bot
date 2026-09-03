"""Persistent checkpoint & production-model management.

Five things are kept deliberately separate:

1. **architecture / source code** — this package (``rl/network.py``, ``rl/agent.py``).
2. **trained weights** — ``checkpoints/`` (+ the curated ``models/``).
3. **training state** — optimizer / counters / RNG, carried *inside* a checkpoint.
4. **training data** — ``data/games.jsonl`` (web-app games), replay buffers (transient).
5. **frontend / UI state** — ``web/`` and ``webapp_state/`` only.

Editing, rebuilding or restarting the frontend touches only (5); it can never
reset, overwrite or retrain (2)/(3).

Directory layout (see ``docs/training-and-models.md``)::

    checkpoints/
      initial/     v000_initial.pt        the adoption baseline (committed)
      experiments/ vNNN.pt                 training / candidate outputs (gitignored)
      production/  best.pt  latest.pt      promoted models (committed)
      registry.json                        the active production model (committed)

Nothing here ever writes to ``checkpoints/production/`` or ``registry.json``
except :meth:`Registry.promote`, which ``scripts/promote_model.py`` calls only
after a candidate passes the promotion criterion.
"""
from __future__ import annotations

import json
import random
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from .agent import AgentMeta, DQNAgent, NetworkConfig

#: repo root — ``src/othello_rl/rl/checkpoint.py`` -> parents[3]
REPO_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
REGISTRY_PATH = CHECKPOINTS_DIR / "registry.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _abs(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (REPO_ROOT / p)


# --------------------------------------------------------------------------- #
# checkpoint (weights + full training state)
# --------------------------------------------------------------------------- #
@dataclass
class Checkpoint:
    net_config: NetworkConfig
    state_dict: Dict[str, Any]
    meta: AgentMeta = field(default_factory=AgentMeta)
    optimizer_state: Optional[dict] = None
    scheduler_state: Optional[dict] = None      # no scheduler object today (epsilon is f(step))
    train_step: int = 0
    episode: int = 0
    games_played: int = 0
    train_config: Optional[dict] = None
    seed: Optional[int] = None
    rng_state: Optional[dict] = None
    experiment: Optional[str] = None
    metrics: Optional[dict] = None
    timestamp: Optional[str] = None
    version: Optional[str] = None
    parent: Optional[str] = None
    net_kind: str = "dqn"                        # "dqn" | "policy_value"
    pv_config: Optional[dict] = None            # set when net_kind == "policy_value"
    format: int = 2

    def build_agent(self, device: str = "cpu"):
        if self.net_kind == "policy_value":
            from .az_agent import PolicyValueAgent
            return PolicyValueAgent.from_state(self.pv_config, self.state_dict, device=device)
        agent = DQNAgent(self.net_config, device=device)
        agent.net.load_state_dict(self.state_dict)
        agent.net.eval()
        agent.meta = self.meta
        return agent


def capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }


def restore_rng_state(state: Optional[dict]) -> None:
    if not state:
        return
    if state.get("python") is not None:
        random.setstate(_as_tuple(state["python"]))
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("torch") is not None:
        t = state["torch"]
        if not isinstance(t, torch.Tensor):
            t = torch.as_tensor(t, dtype=torch.uint8)
        torch.set_rng_state(t.to(torch.uint8))


def _as_tuple(obj):
    """random.setstate needs nested tuples; torch.save round-trips them as lists."""
    if isinstance(obj, list):
        return tuple(_as_tuple(x) for x in obj)
    return obj


def save_checkpoint(path, agent: DQNAgent, *, optimizer=None, train_step: Optional[int] = None,
                    episode: Optional[int] = None, games_played: Optional[int] = None,
                    train_config: Optional[dict] = None, seed: Optional[int] = None,
                    rng_state: Optional[dict] = None, experiment: Optional[str] = None,
                    metrics: Optional[dict] = None, version: Optional[str] = None,
                    parent: Optional[str] = None) -> Path:
    """Write a full checkpoint (weights + training state) to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(
        path,
        optimizer_state=optimizer.state_dict() if optimizer is not None else None,
        train_step=int(train_step if train_step is not None else agent.meta.train_steps),
        episode=int(episode if episode is not None else agent.meta.episodes),
        games_played=int(games_played or 0),
        train_config=train_config,
        seed=seed,
        rng_state=rng_state,
        experiment=experiment,
        metrics=metrics,
        timestamp=_now(),
        version=version,
        parent=parent,
    )
    return path


def load_checkpoint(path) -> Checkpoint:
    """Read a checkpoint written by :func:`save_checkpoint` **or** the older
    ``format: 1`` :meth:`DQNAgent.save` (missing fields default sensibly)."""
    raw = torch.load(_abs(path), map_location="cpu", weights_only=False)
    m = raw.get("meta", {}) or {}
    meta = AgentMeta(train_steps=m.get("train_steps", 0), env_steps=m.get("env_steps", 0),
                     episodes=m.get("episodes", 0), extra=dict(m.get("extra", {})))
    net_kind = raw.get("net_kind", "dqn")
    net_config = (NetworkConfig(**raw["net_config"]) if raw.get("net_config")
                  else NetworkConfig())
    return Checkpoint(
        net_config=net_config,
        state_dict=raw["state_dict"],
        meta=meta,
        net_kind=net_kind,
        pv_config=raw.get("pv_config"),
        optimizer_state=raw.get("optimizer_state"),
        scheduler_state=raw.get("scheduler_state"),
        train_step=int(raw.get("train_step", meta.train_steps)),
        episode=int(raw.get("episode", meta.episodes)),
        games_played=int(raw.get("games_played", meta.extra.get("games_finetuned", 0))),
        train_config=raw.get("train_config"),
        seed=raw.get("seed"),
        rng_state=raw.get("rng_state"),
        experiment=raw.get("experiment"),
        metrics=raw.get("metrics"),
        timestamp=raw.get("timestamp"),
        version=raw.get("version", meta.extra.get("version")),
        parent=raw.get("parent", meta.extra.get("parent")),
        format=int(raw.get("format", 1)),
    )


def load_agent(path, device: str = "cpu"):
    """Load whatever kind of net a checkpoint holds as a playable ``Agent``
    (``DQNAgent`` or ``PolicyValueAgent``)."""
    return load_checkpoint(path).build_agent(device=device)


def save_policy_value_checkpoint(path, net, pv_config, *, optimizer=None,
                                 epoch: int = 0, train_config: Optional[dict] = None,
                                 seed: Optional[int] = None, rng_state: Optional[dict] = None,
                                 metrics: Optional[dict] = None, experiment: Optional[str] = None,
                                 version: Optional[str] = None, parent: Optional[str] = None,
                                 dataset_version: Optional[str] = None,
                                 games_played: int = 0) -> Path:
    """Write a ``net_kind: "policy_value"`` checkpoint (supervised pretraining)."""
    from dataclasses import asdict as _asdict
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pv = pv_config if isinstance(pv_config, dict) else _asdict(pv_config)
    torch.save({
        "format": 2, "net_kind": "policy_value", "pv_config": pv,
        "state_dict": net.state_dict(),
        "meta": {"train_steps": int(epoch), "episodes": 0, "env_steps": 0,
                 "extra": {"epoch": int(epoch), "dataset_version": dataset_version,
                           "method": "supervised_pretrain", "version": version,
                           "parent": parent}},
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "train_step": int(epoch), "episode": 0, "games_played": int(games_played),
        "train_config": train_config, "seed": seed, "rng_state": rng_state,
        "experiment": experiment, "metrics": metrics, "timestamp": _now(),
        "version": version, "parent": parent, "dataset_version": dataset_version,
    }, path)
    return path


@dataclass
class ResumeState:
    env_steps: int
    train_steps: int
    episodes: int
    optimizer_state: Optional[dict]
    rng_state: Optional[dict]
    seed: Optional[int]
    version: Optional[str]
    net_config: NetworkConfig
    source: str


def restore_training(path, agent: DQNAgent, optimizer=None) -> ResumeState:
    """Load ``path`` into ``agent`` (weights) and ``optimizer`` (Adam state),
    returning the counters / RNG the trainer should adopt.

    The replay buffer is intentionally *not* restored — it re-warms from fresh
    rollouts (documented in ``docs/training-and-models.md``). ``agent.net_config``
    is taken from the checkpoint so the architecture always matches the weights.
    """
    ckpt = load_checkpoint(path)
    agent.net_config = ckpt.net_config
    agent.net = ckpt.net_config.build().to(agent.device)
    agent.net.load_state_dict(ckpt.state_dict)
    agent.meta = ckpt.meta
    if optimizer is not None and ckpt.optimizer_state is not None:
        optimizer.load_state_dict(ckpt.optimizer_state)
    return ResumeState(
        env_steps=ckpt.meta.env_steps, train_steps=ckpt.train_step,
        episodes=ckpt.episode, optimizer_state=ckpt.optimizer_state,
        rng_state=ckpt.rng_state, seed=ckpt.seed, version=ckpt.version,
        net_config=ckpt.net_config, source=str(_abs(path)),
    )


# --------------------------------------------------------------------------- #
# resolving a checkpoint spec
# --------------------------------------------------------------------------- #
def resolve_checkpoint(spec: str, root=CHECKPOINTS_DIR) -> Path:
    """Turn a user-facing spec into a concrete ``.pt`` path.

    ``"latest"`` / ``"best"`` -> ``production/{latest,best}.pt``;
    ``"vNNN"`` / ``"v000_initial"`` -> a match under ``experiments/`` or ``initial/``;
    an existing run directory -> ``<run>/checkpoints/final.pt``;
    anything else is treated as a path (repo-root relative if not absolute).
    """
    root = Path(root)
    s = str(spec).strip()
    if s in ("latest", "best"):
        return root / "production" / f"{s}.pt"
    if s.startswith("v") and "/" not in s and not s.endswith(".pt"):
        for sub in ("experiments", "initial", "production"):
            hit = sorted((root / sub).glob(f"{s}*.pt"))
            if hit:
                return hit[0]
        return root / "experiments" / f"{s}.pt"
    p = _abs(s)
    if p.is_dir():
        return p / "checkpoints" / "final.pt"
    return p


def next_experiment_version(root=CHECKPOINTS_DIR) -> str:
    exp = Path(root) / "experiments"
    n = 0
    for f in exp.glob("v*.pt"):
        try:
            n = max(n, int(f.stem[1:].split("_")[0]))
        except ValueError:
            continue
    return f"v{n + 1:03d}"


# --------------------------------------------------------------------------- #
# active production-model registry
# --------------------------------------------------------------------------- #
_DEFAULT_REGISTRY = {
    "model_version": "v000_initial",
    "checkpoint": "checkpoints/initial/v000_initial.pt",
    "parent": None,
    "method": "adopted",
    "training_games": 0,
    "trained_env_steps": 0,
    "evaluation": {},
    "promotion_criterion": (
        "wilson_lb(win_rate vs current best) > 0.5 AND "
        "win_rate vs random >= prev - slack AND win_rate vs greedy >= prev - slack"
    ),
    "promoted_at": None,
    "seed": None,
}


class Registry:
    """The single source of truth for *which* checkpoint the backend serves."""

    def __init__(self, data: dict, path: Path = REGISTRY_PATH, exists: bool = True):
        self.data = data
        self.path = Path(path)
        self._exists = exists

    # -- io ----------------------------------------------------------
    @classmethod
    def load(cls, path=REGISTRY_PATH) -> "Registry":
        path = _abs(path)
        if path.is_file():
            return cls(json.loads(path.read_text()), path=path, exists=True)
        return cls(dict(_DEFAULT_REGISTRY), path=path, exists=False)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2) + "\n")
        self._exists = True

    # -- queries ---------------------------------------------------
    def is_default(self) -> bool:
        """True when no registry file exists on disk (serving the initial model)."""
        return not self._exists

    @property
    def model_version(self) -> str:
        return self.data.get("model_version", "v000_initial")

    def active_checkpoint_path(self) -> Path:
        return _abs(self.data.get("checkpoint", _DEFAULT_REGISTRY["checkpoint"]))

    # -- the only writer of production/ ---------------------------
    def promote(self, candidate_path, *, version: str, parent: Optional[str],
                method: str, evaluation: dict, training_games: int = 0,
                trained_env_steps: int = 0, seed: Optional[int] = None,
                forced: bool = False) -> dict:
        """Copy ``candidate_path`` to ``production/best.pt`` + ``latest.pt`` and
        rewrite ``registry.json``. Callers must have already checked the
        promotion criterion (``scripts/promote_model.py``)."""
        candidate_path = _abs(candidate_path)
        prod = self.path.parent / "production"
        prod.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(candidate_path, prod / "best.pt")
        shutil.copyfile(candidate_path, prod / "latest.pt")
        self.data = {
            "model_version": version,
            "checkpoint": "checkpoints/production/best.pt",
            "parent": parent,
            "method": method,
            "training_games": int(training_games),
            "trained_env_steps": int(trained_env_steps),
            "evaluation": dict(evaluation),
            "promotion_criterion": _DEFAULT_REGISTRY["promotion_criterion"],
            "promoted_at": _now(),
            "seed": seed,
            "forced": bool(forced),
        }
        self.save()
        return dict(self.data)

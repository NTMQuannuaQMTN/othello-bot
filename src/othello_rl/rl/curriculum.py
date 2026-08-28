"""Run a multi-stage curriculum of DQN training against fixed opponents,
with periodic evaluation, checkpointing and metric logging.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from othello_rl.evaluation.harness import evaluate_agent, flatten_eval
from othello_rl.utils.logging import MetricLogger
from .agent import DQNAgent
from .opponents import FixedOpponentEnv
from .trainer import DQNConfig, DQNTrainer


@dataclass
class Stage:
    name: str
    opponent: Union[str, List[str]]          # spec or list of specs (sampled per episode)
    env_steps: int
    learner_color: str = "random"            # BLACK / WHITE / "random"


@dataclass
class CurriculumConfig:
    stages: List[Stage]
    eval_opponents: Dict[str, str] = field(default_factory=lambda: {
        "random": "random", "greedy": "greedy", "heuristic": "heuristic",
    })
    eval_games: int = 100
    eval_every: int = 5_000
    eval_seed: int = 12345
    checkpoint_every: int = 10_000
    dqn: DQNConfig = field(default_factory=DQNConfig)


def _make_eval_fn(agent, cfg: CurriculumConfig, logger: MetricLogger, stage_name: str):
    def eval_fn(trainer: DQNTrainer) -> Dict[str, float]:
        agent.net.eval()
        result = evaluate_agent(agent, cfg.eval_opponents, num_games=cfg.eval_games,
                                seed=cfg.eval_seed)
        flat = flatten_eval(result)
        row = {"phase": "eval", "stage": stage_name, "train_steps": trainer.train_steps, **flat}
        logger.log(**row)
        return {"phase": "eval", "stage": stage_name, **flat}
    return eval_fn


def run_curriculum(agent: DQNAgent, cfg: CurriculumConfig, run_dir: str | Path,
                   seed: int = 0, progress: bool = False) -> MetricLogger:
    run_dir = Path(run_dir)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger = MetricLogger(run_dir / "metrics.jsonl")

    # baseline: evaluate the untrained network first (the comparison anchor)
    base = evaluate_agent(agent, cfg.eval_opponents, num_games=cfg.eval_games, seed=cfg.eval_seed)
    logger.log(phase="eval", stage="<untrained>", env_steps=0, train_steps=0,
               **flatten_eval(base))
    agent.save(ckpt_dir / "untrained.pt")

    trainer: Optional[DQNTrainer] = None
    total_env_steps = 0
    for stage_idx, stage in enumerate(cfg.stages):
        env = FixedOpponentEnv(stage.opponent, learner_color=stage.learner_color,
                               seed=seed + 101 * (stage_idx + 1))
        if trainer is None:
            trainer = DQNTrainer(env, agent, cfg.dqn, seed=seed)
        else:
            trainer.env = env
            trainer._obs, trainer._info = env.reset(seed=seed)

        eval_fn = _make_eval_fn(agent, cfg, logger, stage.name)
        last_ckpt = [0]

        def periodic(tr: DQNTrainer, _stage=stage, _last=last_ckpt):
            if tr.env_steps - _last[0] >= cfg.checkpoint_every:
                _last[0] = tr.env_steps
                tr._sync_agent_meta()
                agent.save(ckpt_dir / f"{_stage.name}_step{tr.env_steps}.pt")

        # drive the trainer in eval-sized chunks so we can also checkpoint
        target = trainer.env_steps + stage.env_steps
        while trainer.env_steps < target:
            chunk = min(cfg.eval_every, target - trainer.env_steps)
            trainer.learn(total_env_steps=chunk, eval_fn=eval_fn,
                          eval_every=cfg.eval_every, log_every=max(500, cfg.eval_every // 5),
                          progress=progress)
            periodic(trainer)
            logger.log(phase="train", stage=stage.name, env_steps=trainer.env_steps,
                       train_steps=trainer.train_steps, episodes=trainer.episodes,
                       epsilon=cfg.dqn.epsilon(trainer.env_steps),
                       mean_return_100=_mean_return(trainer))

        trainer._sync_agent_meta()
        agent.save(ckpt_dir / f"{stage.name}_final.pt")
        total_env_steps = trainer.env_steps

    agent.save(ckpt_dir / "final.pt")
    logger.log(phase="done", env_steps=total_env_steps)
    return logger


def _mean_return(trainer: DQNTrainer) -> float:
    import numpy as np
    return float(np.mean(trainer._returns)) if trainer._returns else 0.0

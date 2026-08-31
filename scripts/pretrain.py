#!/usr/bin/env python3
"""Supervised imitation pretraining of a policy(+value) net on historical games.

    python3 scripts/build_dataset.py --config configs/dataset.yaml     # -> <version>
    python3 scripts/pretrain.py --config configs/pretrain.yaml --dataset <version>
    python3 scripts/pretrain.py --config configs/pretrain.yaml --dataset <version> \
        --resume checkpoints/experiments/<v>.pt

Writes checkpoints/experiments/<v>.pt (net_kind: policy_value), an
experiments/<ts>_pretrain/ run dir with metrics.jsonl + metadata.json, and one
row in experiments/index.jsonl. NEVER touches checkpoints/production/.
This is behaviour cloning, not RL.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_ROOT = Path(__file__).resolve().parents[1]

from othello_rl.rl.checkpoint import (next_experiment_version,  # noqa: E402
                                      save_policy_value_checkpoint)
from othello_rl.rl.supervised import SupervisedConfig, SupervisedTrainer  # noqa: E402
from othello_rl.utils.config import load_config  # noqa: E402
from othello_rl.utils.experiment import (create_run_dir, log_experiment,  # noqa: E402
                                         write_metadata)
from othello_rl.utils.logging import MetricLogger  # noqa: E402
from othello_rl.utils.seed import seed_everything  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(_ROOT / "configs" / "pretrain.yaml"))
    ap.add_argument("--dataset", required=True, help="training_data/<version> dir or name")
    ap.add_argument("--resume", default=None, help="a policy_value checkpoint to continue")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    raw = dict(load_config(args.config))
    cfg = SupervisedConfig(**{k: raw[k] for k in raw
                              if k in SupervisedConfig.__dataclass_fields__})
    seed_everything(cfg.seed)

    ds_dir = Path(args.dataset)
    if not ds_dir.is_dir():
        ds_dir = _ROOT / "data" / "processed" / "training_data" / args.dataset
    train_npz, val_npz = ds_dir / "train.npz", ds_dir / "val.npz"
    if not train_npz.is_file():
        print(f"ERROR: {train_npz} not found — run scripts/build_dataset.py first",
              file=sys.stderr)
        return 2

    if args.resume:
        trainer = SupervisedTrainer.resume(cfg, args.resume, device=args.device)
        parent = args.resume
        print(f"resuming {args.resume} @ epoch {trainer.epoch}")
    else:
        trainer = SupervisedTrainer(cfg, device=args.device)
        parent = None

    run_dir = create_run_dir(str(_ROOT / "experiments"), "pretrain")
    mlog = MetricLogger(run_dir / "metrics.jsonl")
    write_metadata(run_dir, dict(raw), extra={"dataset": ds_dir.name, "resume": args.resume})
    print(f"run dir: {run_dir}")

    def on_epoch(m):
        mlog.log(**{k: getattr(m, k) for k in m.__dataclass_fields__})
        print(f"  epoch {m.epoch:>3}  train {m.train_loss:.4f}  val {m.val_loss:.4f}  "
              f"move-acc {m.move_accuracy:.3f}  ({m.seconds}s)")

    history = trainer.fit(train_npz, val_npz, on_epoch=on_epoch)
    last = history[-1]

    version = next_experiment_version()
    out = _ROOT / "checkpoints" / "experiments" / f"{version}_pretrain.pt"
    save_policy_value_checkpoint(
        out, trainer.net, cfg.net_config(), optimizer=trainer.opt, epoch=trainer.epoch,
        train_config=dict(raw), seed=cfg.seed, rng_state=trainer.rng_state(),
        metrics={k: getattr(last, k) for k in ("val_loss", "move_accuracy")},
        experiment=run_dir.name, version=version, parent=parent,
        dataset_version=ds_dir.name)
    print(f"\ncheckpoint -> {out}")

    log_experiment({
        "kind": "pretrain", "experiment": run_dir.name, "parent": parent,
        "dataset_version": ds_dir.name, "epochs": trainer.epoch, "seed": cfg.seed,
        "arch": cfg.az_network, "final_train_loss": last.train_loss,
        "val_loss": last.val_loss, "move_accuracy": last.move_accuracy,
        "checkpoint": str(out.relative_to(_ROOT)), "promotion": "pending",
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

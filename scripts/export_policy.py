#!/usr/bin/env python3
"""Export a trained checkpoint to a **torch-free** ``.npz`` for the web deploy.

The Vercel deployment serves moves/grades with :class:`NumpyPolicy` (numpy only,
no PyTorch — see ``src/othello_rl/rl/numpy_policy.py``).  Run this once, with the
full install, to produce the weights it loads:

    python3 scripts/export_policy.py                      # -> web/api/policy.npz
    python3 scripts/export_policy.py --checkpoint <path>

Also re-vendors ``src/othello_rl`` -> ``web/api/othello_rl`` so ``web/api/`` is a
self-contained function bundle.  Both are committed (~2 MB total).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import torch  # noqa: E402

from othello_rl.rl.checkpoint import Registry, resolve_checkpoint  # noqa: E402
from othello_rl.rl.numpy_policy import NumpyPolicy  # noqa: E402
from othello_rl.environment.environment import encode_observation, legal_action_mask  # noqa: E402
from othello_rl.environment.board import Board  # noqa: E402
from othello_rl.rl.agent import DQNAgent  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=None, help="default: active production checkpoint")
    ap.add_argument("--out", default="web/api", help="directory to write policy.npz into")
    args = ap.parse_args(argv)

    ckpt = (Path(args.checkpoint) if args.checkpoint and Path(args.checkpoint).exists()
            else resolve_checkpoint(args.checkpoint) if args.checkpoint
            else Registry.load().active_checkpoint_path())
    raw = torch.load(Path(ckpt), map_location="cpu", weights_only=False)
    sd = raw["state_dict"]
    cfg = raw["net_config"]
    meta = raw.get("meta", {})

    arrays = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in sd.items()}
    arrays["net_config_json"] = np.array(json.dumps(cfg))
    arrays["meta_json"] = np.array(json.dumps({
        "train_steps": meta.get("train_steps", 0),
        "env_steps": meta.get("env_steps", 0),
        "episodes": meta.get("episodes", 0),
        "extra": meta.get("extra", {}),
        "source_checkpoint": str(ckpt),
    }))

    out_dir = Path(args.out) if Path(args.out).is_absolute() else _ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / "policy.npz"
    np.savez_compressed(npz, **arrays)

    # sanity: numpy forward must match torch on a batch of real positions
    tagent = DQNAgent.from_checkpoint(str(ckpt))
    npol = NumpyPolicy.load(npz)
    rng = np.random.default_rng(0)
    max_err = 0.0
    for _ in range(40):
        b = Board.initial()
        for _ in range(int(rng.integers(0, 30))):
            legal = b.legal_moves()
            if not legal:
                b = b.apply(None)
                continue
            b = b.apply(legal[int(rng.integers(0, len(legal)))])
        if b.is_terminal() or not b.legal_moves():
            continue
        obs, msk = encode_observation(b), legal_action_mask(b)
        qt = tagent.q_values(obs, msk)
        qn = npol.q_values(obs, msk)
        finite = np.isfinite(qt) & np.isfinite(qn)
        max_err = max(max_err, float(np.abs(qt[finite] - qn[finite]).max()))
        assert tagent.greedy_act(obs, msk) == npol.greedy_act(obs, msk), "argmax mismatch"

    size_kb = npz.stat().st_size / 1024
    print(f"wrote {npz}  ({size_kb:.0f} KB, {npol.param_count:,} params)")
    print(f"numpy vs torch: max |ΔQ| = {max_err:.2e}, argmax matches on all sampled positions")

    # re-vendor the package so web/api/ is a self-contained function bundle
    import shutil
    vendor = _ROOT / "web" / "api" / "othello_rl"
    shutil.rmtree(vendor, ignore_errors=True)
    shutil.copytree(_ROOT / "src" / "othello_rl", vendor,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    print(f"vendored {vendor}  ({sum(f.stat().st_size for f in vendor.rglob('*') if f.is_file()) // 1024} KB)")
    print("commit web/api/policy.npz + web/api/othello_rl/ for the Vercel deploy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

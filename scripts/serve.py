#!/usr/bin/env python3
"""Run the Othello bot JSON API (play + Lichess-style analysis). Inference only —
training is offline (``scripts/train_*.py``).

The React front end is in ``web/``:

    # dev (hot reload; Vite proxies /api to this server):
    python3 scripts/serve.py --config configs/webapp.yaml     # terminal 1
    cd web && npm install && npm run dev                      # terminal 2  -> :5173

    # or build once and let this server serve it:
    cd web && npm run build
    python3 scripts/serve.py                                  # -> http://127.0.0.1:8000

    # mirror the (torch-free) Vercel deploy:
    python3 scripts/serve.py --policy api/policy.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from othello_rl.environment.board import Board  # noqa: E402
from othello_rl.webapp.bot_service import OthelloBot  # noqa: E402
from othello_rl.webapp.server import serve  # noqa: E402


def _rel(p) -> Path:
    """Resolve a path: absolute as-is, else relative to CWD if that exists
    (``npm run api`` runs this from web/ with ``../configs/…``), else relative
    to the repo root."""
    p = Path(p)
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    return _ROOT / p


def _resolve_checkpoint(explicit) -> Path:
    if explicit:
        p = _rel(explicit)
        if p.exists():
            return p
        from othello_rl.rl.checkpoint import resolve_checkpoint
        return Path(resolve_checkpoint(explicit))
    from othello_rl.rl.checkpoint import Registry
    reg = Registry.load()
    if not reg.is_default():
        return Path(reg.active_checkpoint_path())
    for cand in ("checkpoints/production/best.pt", "checkpoints/initial/v000_initial.pt",
                 "models/othello_bot_v1.pt"):
        if _rel(cand).is_file():
            return _rel(cand)
    raise SystemExit("ERROR: no model — register one with scripts/promote_model.py, "
                     "pass --checkpoint, or set 'checkpoint:' in the config.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/webapp.yaml")
    ap.add_argument("--checkpoint", default=None, help="override the bot checkpoint (.pt)")
    ap.add_argument("--policy", default=None,
                    help="serve the torch-free exported policy (api/policy.npz) instead")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--host", default=None)
    args = ap.parse_args(argv)

    cfg = {}
    cfg_path = _rel(args.config)
    if cfg_path.is_file():
        from othello_rl.utils.config import load_config
        cfg = load_config(str(cfg_path))

    if args.policy:
        model_path = _rel(args.policy)
        label = f"torch-free policy {model_path.name}"
    else:
        model_path = _resolve_checkpoint(args.checkpoint or cfg.get("checkpoint"))
        label = str(model_path)
    if not Path(model_path).exists():
        raise SystemExit(f"ERROR: model not found: {model_path}")

    bot = OthelloBot.load(str(model_path))

    opening = bot.select_action(Board.initial())
    legal = {r * 8 + c for r, c in Board.initial().legal_moves()}
    if opening not in legal:
        raise SystemExit(f"ERROR: loaded model returned an illegal opening move ({opening})")

    host = args.host or cfg.get("host", "127.0.0.1")
    port = args.port or int(cfg.get("port", 8000))
    static_dir = _rel(cfg.get("static_dir", "web/dist"))
    httpd = serve(bot, host=host, port=port, static_dir=static_dir)

    i = bot.info()
    built = (static_dir / "index.html").is_file()
    print(f"model: {label}  ({i['params']:,} params, env_steps={i['train_env_steps']:,})\n"
          f"  opening move OK ({opening})\n"
          f"  Othello bot API on http://{host}:{port}\n"
          f"  {'serving web/dist' if built else 'front end not built — run: cd web && npm run build'}\n"
          f"  (Ctrl-C to stop)\n", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
        httpd.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

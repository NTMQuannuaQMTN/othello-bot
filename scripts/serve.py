#!/usr/bin/env python3
"""Run the Othello bot web app (play + fine-tune + Lichess-style analysis).

    python3 scripts/serve.py --config configs/webapp.yaml
    open http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from othello_rl.utils.config import load_config  # noqa: E402
from othello_rl.utils.seed import seed_everything  # noqa: E402
from othello_rl.webapp.bot_service import FineTuneConfig, OthelloBot  # noqa: E402
from othello_rl.webapp.server import serve  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/webapp.yaml")
    ap.add_argument("--checkpoint", default=None, help="override the bot checkpoint")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--host", default=None)
    ap.add_argument("--fresh", action="store_true",
                    help="ignore any fine-tuned state and start from the base checkpoint")
    args = ap.parse_args(argv)
    seed_everything(0)

    cfg = load_config(args.config)
    host = args.host or cfg.get("host", "127.0.0.1")
    port = args.port or int(cfg.get("port", 8000))
    base_ckpt = Path(args.checkpoint or cfg.get("checkpoint", "models/othello_bot_v1.pt"))
    state_dir = Path(cfg.get("state_dir", "webapp_state"))
    state_dir.mkdir(parents=True, exist_ok=True)

    current = state_dir / "current.pt"
    if args.fresh or not current.exists():
        shutil.copyfile(base_ckpt, current)
        print(f"initialised bot state from {base_ckpt}")

    ft = FineTuneConfig(**{k: v for k, v in dict(cfg.get("finetune", {})).items()})
    bot = OthelloBot.load(str(current), source_path=str(base_ckpt),
                          state_dir=str(state_dir), ft_config=ft)
    print(f"bot: {bot.info()}")

    httpd = serve(bot, host=host, port=port)
    url = f"http://{host}:{port}"
    print(f"\n  Othello bot web app running at {url}\n  (Ctrl-C to stop)\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
        httpd.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

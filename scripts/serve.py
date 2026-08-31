#!/usr/bin/env python3
"""Run the Othello bot JSON API (play + fine-tune + Lichess-style analysis).

The React front end is in `web/`:

    # dev (hot reload; proxies /api to this server):
    python3 scripts/serve.py --config configs/webapp.yaml     # terminal 1
    cd web && npm install && npm run dev                      # terminal 2  -> :5173

    # or build once and let this server serve it:
    cd web && npm run build
    python3 scripts/serve.py                                  # -> http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from othello_rl.environment.board import Board  # noqa: E402
from othello_rl.rl.checkpoint import Registry, resolve_checkpoint  # noqa: E402
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

    _root = Path(__file__).resolve().parents[1]

    def _rel(p) -> Path:
        """Resolve a path against the repo root so the server works no matter
        which directory it's launched from (e.g. `npm run api` runs it from web/)."""
        p = Path(p)
        return p if p.is_absolute() else (_root / p)

    cfg = load_config(_rel(args.config) if not Path(args.config).exists() else args.config)
    host = args.host or cfg.get("host", "127.0.0.1")
    port = args.port or int(cfg.get("port", 8000))

    # -- resolve the PRODUCTION checkpoint --------------------------------
    #   --checkpoint  >  configs/webapp.yaml: checkpoint  >  checkpoints/registry.json
    #   >  checkpoints/initial/v000_initial.pt  >  models/othello_bot_v1.pt
    # The backend NEVER builds a random network; a missing file is a hard error.
    reg = Registry.load()
    origin = None
    if args.checkpoint:
        base_ckpt = Path(args.checkpoint)
        if not base_ckpt.exists():
            base_ckpt = resolve_checkpoint(args.checkpoint)
        origin = "--checkpoint"
    elif cfg.get("checkpoint"):
        base_ckpt = _rel(cfg["checkpoint"])
        origin = "configs/webapp.yaml"
    elif not reg.is_default():
        base_ckpt = reg.active_checkpoint_path()
        origin = f"registry ({reg.model_version})"
    else:
        for cand in ("checkpoints/initial/v000_initial.pt", "models/othello_bot_v1.pt"):
            if _rel(cand).is_file():
                base_ckpt = _rel(cand)
                origin = f"default ({cand})"
                break
        else:
            print("ERROR: no production model — register one with "
                  "scripts/promote_model.py or set 'checkpoint:' in the config.",
                  file=sys.stderr)
            return 2

    state_dir = _rel(cfg.get("state_dir", "webapp_state"))
    state_dir.mkdir(parents=True, exist_ok=True)
    static_dir = _rel(cfg.get("static_dir", "web/dist"))
    games_path = _rel(cfg.get("games_path", "data/games.jsonl"))

    if not base_ckpt.is_file():
        print(f"ERROR: bot checkpoint not found: {base_ckpt} (from {origin})", file=sys.stderr)
        return 2
    if "initial" in str(base_ckpt) or "othello_bot_v1" in str(base_ckpt):
        if reg.is_default():
            print("WARNING: no registered production model — serving the INITIAL "
                  "checkpoint (not fine-tuned). Promote one with scripts/promote_model.py.")

    current = state_dir / "current.pt"
    if args.fresh:
        shutil.copyfile(base_ckpt, current)
        print(f"--fresh: discarded webapp_state/current.pt, re-seeded from {base_ckpt}")
    elif not current.exists():
        shutil.copyfile(base_ckpt, current)
        print(f"initialised scratch bot state from {base_ckpt}")

    ft = FineTuneConfig(**{k: v for k, v in dict(cfg.get("finetune", {})).items()})
    bot = OthelloBot.load(str(current), source_path=str(base_ckpt),
                          state_dir=str(state_dir), ft_config=ft)

    # -- startup verification: the model must make a legal opening move ---
    opening = bot.select_action(Board.initial())
    legal = {r * 8 + c for r, c in Board.initial().legal_moves()}
    if opening not in legal:
        print(f"ERROR: loaded model returned an illegal opening move ({opening})",
              file=sys.stderr)
        return 3
    _i = bot.info()
    print(f"model: v{_i['version']} (parent {_i['parent']}) from {base_ckpt} [{origin}]"
          f"\n  {_i['params']} params · env_steps={_i['train_env_steps']} "
          f"· games_finetuned={_i['games_finetuned']} · baseline={_i['baseline']}"
          f"\n  opening move OK ({opening})")

    httpd = serve(bot, host=host, port=port, static_dir=static_dir,
                  games_path=str(games_path))
    print(f"game dataset: {games_path}")
    url = f"http://{host}:{port}"
    built = (static_dir / "index.html").is_file()
    print(f"\n  Othello bot API on {url}"
          f"\n  {'serving web/dist' if built else 'front end not built — run: cd web && npm run dev'}"
          f"\n  (Ctrl-C to stop)\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
        httpd.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

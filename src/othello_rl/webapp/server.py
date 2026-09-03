"""Zero-dependency JSON API + static file server for the Othello bot web app.

The API is **stateless**: the Play tab sends its move history with every request
(``{human_color, history_actions}``) so the same routes work behind a long-lived
process *and* behind stateless serverless functions (the Vercel deploy — see
``api/index.py`` and ``docs/deploy.md``).

Endpoints
---------
GET  /                       -> the single-page app
GET  /static/<file>          -> assets
GET  /api/bot | /api/model   -> bot info (version, params, can_finetune)
POST /api/new     {human_color}                 -> fresh game state
POST /api/move    {human_color, history_actions, action, bot_reply?}
POST /api/bot_move {human_color, history_actions}
POST /api/state   {human_color, history_actions}
POST /api/eval    {history_actions}             -> eval bar
POST /api/best_move {history_actions, time_budget?}
POST /api/analyse {moves|transcript|history_actions}
POST /api/finetune / /api/finetune_all / /api/games / /api/bot/reset
                             -> full install only (503 on the inference deploy)
"""
from __future__ import annotations

import json
import threading
import time
import traceback
from dataclasses import asdict, is_dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

from othello_rl.environment.board import Board, PASS_ACTION

from .bot_service import OthelloBot
from .moves import parse_game
from .session import GameSession


def _board_from(actions) -> Board:
    """Replay a flat action list (64 = pass) from the start position."""
    b = Board.initial()
    for a in (int(x) for x in actions):
        b = b.apply(None if (a == PASS_ACTION or not b.legal_moves()) else divmod(a, 8))
    return b


#: Vite build output (``cd web && npm run build``). Overridable via AppState.
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STATIC_DIR = _REPO_ROOT / "web" / "dist"

_CONTENT_TYPES = {".html": "text/html", ".js": "application/javascript",
                  ".mjs": "application/javascript", ".css": "text/css",
                  ".json": "application/json", ".svg": "image/svg+xml",
                  ".ico": "image/x-icon", ".png": "image/png", ".woff2": "font/woff2",
                  ".map": "application/json"}

_FALLBACK_PAGE = b"""<!doctype html><meta charset=utf-8>
<title>OthelloRL API</title>
<style>body{font:15px/1.5 system-ui;margin:40px;max-width:640px;color:#222}code{background:#eee;padding:2px 5px;border-radius:4px}</style>
<h1>Othello<span style=color:#6ea8fe>RL</span> &mdash; API is running</h1>
<p>The React front end hasn't been built yet. Either:</p>
<pre><code>cd web && npm install && npm run dev</code></pre>
<p>&hellip;and open the Vite dev server it prints (it proxies <code>/api</code> here), or</p>
<pre><code>cd web && npm run build</code></pre>
<p>&hellip;then reload this page (this server serves <code>web/dist</code>).</p>
<p>The JSON API itself is live now &mdash; e.g. <a href="/api/bot">/api/bot</a>.</p>
"""


class AppState:
    def __init__(self, bot: OthelloBot, static_dir: Path = None, games_path=None):
        self.bot = bot
        self.lock = threading.Lock()
        self.static_dir = Path(static_dir) if static_dir else DEFAULT_STATIC_DIR
        # the durable game dataset (append-only, deduplicated by move sequence).
        # None on a read-only / serverless deploy — save + auto-record no-op then.
        if games_path is not None:
            self.games_path = Path(games_path)
        elif bot.state_dir:
            self.games_path = Path(bot.state_dir) / "games.jsonl"
        else:
            self.games_path = None
        self._seen = set()      # move-sequence signatures already on disk
        if self.games_path and self.games_path.is_file():
            for g in self.load_games():
                self._seen.add(tuple(g.get("moves", [])))

    # -- stateless game sessions --------------------------------------
    def session_from(self, body: dict) -> GameSession:
        """Rebuild a play session from the request body (stateless)."""
        return GameSession.from_history(
            self.bot,
            human_color=body.get("human_color", "black"),
            history_actions=body.get("history_actions", []),
        )

    def _append_game(self, moves, human_color="black", learn_color=None) -> dict:
        if not self.games_path:
            return {"saved": False, "count": 0, "reason": "dataset not available on this deployment"}
        moves = [int(a) for a in moves]
        if not moves:
            return {"saved": False, "count": len(self.load_games()), "reason": "empty game"}
        sig = tuple(moves)
        if sig in self._seen:
            return {"saved": False, "count": len(self.load_games()), "reason": "already saved"}
        board = Board.initial()
        for a in moves:
            mv = None if a == PASS_ACTION or not board.legal_moves() else divmod(a, 8)
            board = board.apply(mv)
        b, w = board.scores()
        self._seen.add(sig)
        rec = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "human_color": human_color,
            "moves": moves,
            "winner": ("black" if board.winner() == 1
                       else "white" if board.winner() == -1 else "draw"),
            "score": {"black": b, "white": w},
            "bot_version": self.bot.version,
        }
        if learn_color:
            rec["learn_color"] = learn_color
        self.games_path.parent.mkdir(parents=True, exist_ok=True)
        with self.games_path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        return {"saved": True, "count": len(self.load_games())}

    def record_if_finished(self, session: GameSession) -> None:
        if not self.games_path:
            return
        st = session
        if not st.board.is_terminal() or not st.history:
            return
        self._append_game(list(st.history),
                          human_color="black" if st.human_color == 1 else "white")

    def load_games(self) -> list:
        if not self.games_path or not self.games_path.is_file():
            return []
        return [json.loads(ln) for ln in self.games_path.read_text().splitlines() if ln.strip()]


def build_routes(app: AppState) -> Dict[str, Callable[[dict], Any]]:
    routes: Dict[str, Callable[[dict], Any]] = {}

    def route(*method_paths):
        def deco(fn):
            for mp in method_paths:
                routes[mp] = fn
            return fn
        return deco

    def _needs_full_install():
        if not app.bot.can_finetune:
            raise ValueError("fine-tuning / dataset editing is not available on this "
                             "deployment (inference-only build).")

    @route("GET /api/bot", "GET /api/model")
    def _bot_info(_):
        d = dict(app.bot.info())
        d["dataset"] = str(app.games_path) if app.games_path else None
        d["dataset_games"] = len(app.load_games())
        d["features"] = {"finetune": app.bot.can_finetune,
                         "dataset": app.games_path is not None}
        return d

    @route("POST /api/new")
    def _new(body):
        hc = str(body.get("human_color", "black")).lower()
        if hc.startswith("r"):
            import random
            hc = random.choice(["black", "white"])
        s = GameSession.from_history(app.bot, hc, [])
        if not s.board.is_terminal() and s.board.player != s.human_color:
            s._bot_turn()
        return s.state()

    @route("GET /api/state", "POST /api/state")
    def _state(body):
        return app.session_from(body or {}).state()

    @route("GET /api/eval", "POST /api/eval")
    def _eval(body):
        acts = (body or {}).get("history_actions")
        b = _board_from(acts) if acts is not None else Board.initial()
        return app.bot.bar_eval(b)

    @route("POST /api/best_move")
    def _best_move(body):
        b = _board_from(body.get("history_actions") or [])
        return app.bot.best_move(b, time_budget=float(body.get("time_budget", 3.0)),
                                 endgame_empties=int(body.get("endgame_empties", 16)))

    @route("POST /api/move")
    def _move(body):
        s = app.session_from(body)
        st = s.human_move(int(body["action"]), bot_reply=body.get("bot_reply", True))
        app.record_if_finished(s)
        return st

    @route("POST /api/bot_move", "GET /api/bot_move")
    def _bot_move(body):
        s = app.session_from(body or {})
        st = s.bot_move()
        app.record_if_finished(s)
        return st

    @route("POST /api/analyse")
    def _analyse(body):
        actions = parse_game(body if body else {})
        return app.bot.analyse_line(actions, top_k=int((body or {}).get("top_k", 3)))

    @route("GET /api/games")
    def _games(_):
        return {"count": len(app.load_games()),
                "path": str(app.games_path) if app.games_path else None}

    @route("POST /api/games")
    def _save_game(body):
        _needs_full_install()
        moves = parse_game(body)
        res = app._append_game(moves, human_color=body.get("human_color", "black"),
                               learn_color=body.get("learn_color"))
        res["path"] = str(app.games_path) if app.games_path else None
        return res

    @route("POST /api/finetune")
    def _finetune(body):
        _needs_full_install()
        actions = parse_game(body)
        learn_color = body.get("learn_color") or "both"
        report = app.bot.finetune_from_game(actions, learn_color)
        d = asdict(report)
        d["learn_color"] = learn_color
        return d

    @route("POST /api/finetune_all")
    def _finetune_all(_):
        _needs_full_install()
        games = app.load_games()
        if not games:
            raise ValueError("no saved games yet — play a few games first")
        report = app.bot.finetune_from_games(games)
        d = asdict(report)
        d["n_games"] = len(games)
        return d

    @route("POST /api/bot/reset")
    def _reset(_):
        _needs_full_install()
        app.bot.reset_to_baseline()
        return app.bot.info()

    return routes


def dispatch(app: AppState, routes: Dict[str, Callable], method: str, path: str,
             body: dict) -> Tuple[int, Any]:
    """Run one API request. Returns ``(status, json-able)``. Framework-free so
    both the stdlib server and the Vercel handler can call it."""
    handler = routes.get(f"{method} {path}")
    if handler is None:
        return 404, {"error": f"no route {method} {path}"}
    try:
        with app.lock:
            return 200, handler(body or {})
    except (ValueError, KeyError) as e:
        return 400, {"error": str(e)}
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return 500, {"error": f"{type(e).__name__}: {e}"}


def make_handler(app: AppState):
    routes = build_routes(app)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send_json(self, obj, status=200):
            payload = json.dumps(obj, default=_json_default).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _send_bytes(self, data: bytes, content_type: str, status: int = 200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(data)

        def _serve_static(self, url_path: str):
            static_dir = app.static_dir
            index = static_dir / "index.html"
            if not index.is_file():
                self._send_bytes(_FALLBACK_PAGE, "text/html")
                return
            rel = url_path.lstrip("/") or "index.html"
            target = (static_dir / rel).resolve()
            if static_dir.resolve() not in target.parents and target != index.resolve():
                target = index
            if not target.is_file():
                target = index
            self._send_bytes(
                target.read_bytes(),
                _CONTENT_TYPES.get(target.suffix, "application/octet-stream"),
            )

        def _dispatch(self, method: str):
            path = self.path.split("?", 1)[0]
            if method == "GET" and not path.startswith("/api/"):
                self._serve_static(path)
                return
            body = {}
            if method == "POST":
                length = int(self.headers.get("Content-Length", 0) or 0)
                raw = self.rfile.read(length) if length else b""
                if raw:
                    try:
                        body = json.loads(raw)
                    except json.JSONDecodeError:
                        self._send_json({"error": "invalid JSON body"}, 400)
                        return
            status, result = dispatch(app, routes, method, path, body)
            self._send_json(result, status)

        def do_GET(self):
            self._dispatch("GET")

        def do_POST(self):
            self._dispatch("POST")

    return Handler


def _json_default(o):
    if is_dataclass(o):
        return asdict(o)
    import numpy as np
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def serve(bot: OthelloBot, host: str = "127.0.0.1", port: int = 8000,
          static_dir=None, games_path=None):
    app = AppState(bot, static_dir=static_dir, games_path=games_path)
    httpd = ThreadingHTTPServer((host, port), make_handler(app))
    return httpd

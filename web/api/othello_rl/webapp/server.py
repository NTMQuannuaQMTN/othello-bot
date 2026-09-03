"""Zero-dependency JSON API + static file server for the Othello bot web app.

Inference only — the bot plays and analyses; training is done offline
(``scripts/train_*.py``).  The API is **stateless**: the Play tab sends its move
history with every request (``{human_color, history_actions}``) so the same
routes work behind a long-lived process *and* behind stateless serverless
functions (the Vercel deploy — see ``api/index.py`` and ``docs/deploy.md``).

Endpoints
---------
GET  /                        -> the single-page app
GET  /api/bot | /api/model    -> bot info (version, params, network)
POST /api/new     {human_color}                     -> fresh game state
POST /api/move    {human_color, history_actions, action, bot_reply?}
POST /api/bot_move {human_color, history_actions}
POST /api/state   {human_color, history_actions}
POST /api/eval    {history_actions}                 -> eval bar
POST /api/best_move {history_actions, time_budget?}
POST /api/analyse {moves|transcript|history_actions}
"""
from __future__ import annotations

import json
import threading
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
<p>The React front end hasn't been built yet:</p>
<pre><code>cd web && npm install && npm run build</code></pre>
<p>&hellip;then reload this page. The JSON API is live now &mdash;
e.g. <a href="/api/bot">/api/bot</a>.</p>
"""


class AppState:
    def __init__(self, bot: OthelloBot, static_dir: Path = None):
        self.bot = bot
        self.lock = threading.Lock()
        self.static_dir = Path(static_dir) if static_dir else DEFAULT_STATIC_DIR

    def session_from(self, body: dict) -> GameSession:
        """Rebuild a play session from the request body (the API is stateless)."""
        return GameSession.from_history(
            self.bot,
            human_color=body.get("human_color", "black"),
            history_actions=body.get("history_actions", []),
        )


def build_routes(app: AppState) -> Dict[str, Callable[[dict], Any]]:
    routes: Dict[str, Callable[[dict], Any]] = {}

    def route(*method_paths):
        def deco(fn):
            for mp in method_paths:
                routes[mp] = fn
            return fn
        return deco

    @route("GET /api/bot", "GET /api/model")
    def _bot_info(_):
        return dict(app.bot.info())

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
        return s.human_move(int(body["action"]), bot_reply=body.get("bot_reply", True))

    @route("POST /api/bot_move", "GET /api/bot_move")
    def _bot_move(body):
        return app.session_from(body or {}).bot_move()

    @route("POST /api/analyse")
    def _analyse(body):
        actions = parse_game(body if body else {})
        return app.bot.analyse_line(actions, top_k=int((body or {}).get("top_k", 3)))

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


def serve(bot: OthelloBot, host: str = "127.0.0.1", port: int = 8000, static_dir=None):
    app = AppState(bot, static_dir=static_dir)
    httpd = ThreadingHTTPServer((host, port), make_handler(app))
    return httpd

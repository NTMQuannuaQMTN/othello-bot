"""Zero-dependency JSON API + static file server for the Othello bot web app.

Endpoints
---------
GET  /                       -> the single-page app
GET  /static/<file>          -> assets
GET  /api/bot                 -> bot info (version, params, games fine-tuned)
GET  /api/state               -> current game state
POST /api/new    {human_color, level}
POST /api/move   {action}     -> apply human move, bot replies
POST /api/bot_move            -> let the bot move (bot plays first / demo)
POST /api/analyse {moves|transcript|history_actions}  -> per-ply analysis
POST /api/finetune {human_color, moves?}   -> fine-tune from the (given / current) game
POST /api/bot/reset           -> restore baseline weights
"""
from __future__ import annotations

import json
import threading
import time
import traceback
from dataclasses import asdict, is_dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict

from .bot_service import OthelloBot
from .moves import parse_game
from .session import GameSession

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
    def __init__(self, bot: OthelloBot, static_dir: Path = None):
        self.bot = bot
        self.session = GameSession(bot)
        self.lock = threading.Lock()
        self.static_dir = Path(static_dir) if static_dir else DEFAULT_STATIC_DIR
        self.games_path = (Path(bot.state_dir) / "games.jsonl") if bot.state_dir else None
        self._recorded = False  # has the current game already been appended?

    def record_if_finished(self) -> None:
        """Append a finished human-vs-bot game to games.jsonl (once)."""
        if not self.games_path or self._recorded:
            return
        st = self.session
        if not st.board.is_terminal() or len(st.history) == 0:
            return
        self._recorded = True
        b, w = st.board.scores()
        rec = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "human_color": "black" if st.human_color == 1 else "white",
            "moves": list(st.history),
            "winner": ("black" if st.board.winner() == 1
                       else "white" if st.board.winner() == -1 else "draw"),
            "score": {"black": b, "white": w},
            "bot_version": self.bot.version,
        }
        with self.games_path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")

    def load_games(self) -> list:
        if not self.games_path or not self.games_path.is_file():
            return []
        return [json.loads(ln) for ln in self.games_path.read_text().splitlines() if ln.strip()]


def make_handler(app: AppState):
    routes: Dict[str, Callable[[dict], Any]] = {}

    def route(method_path):
        def deco(fn):
            routes[method_path] = fn
            return fn
        return deco

    # ---- API handlers ------------------------------------------------
    @route("GET /api/bot")
    def _bot_info(_):
        return app.bot.info()

    @route("GET /api/state")
    def _state(_):
        return app.session.state()

    @route("GET /api/eval")
    def _eval(_):
        """Bot's read of the current game position (for the eval bar)."""
        return app.bot.evaluate_position(app.session.board)

    @route("POST /api/new")
    def _new(body):
        app._recorded = False
        return app.session.new_game(body.get("human_color", "black"),
                                    int(body.get("level", 0)))

    @route("POST /api/move")
    def _move(body):
        # bot_reply=false: apply only the human move (the client then calls
        # /api/bot_move after a short pause so the bot's reply is visible).
        st = app.session.human_move(int(body["action"]),
                                    bot_reply=body.get("bot_reply", True))
        app.record_if_finished()
        return st

    @route("POST /api/bot_move")
    @route("GET /api/bot_move")  # tolerate a GET (no body) from the client
    def _bot_move(_):
        st = app.session.bot_move()
        app.record_if_finished()
        return st

    @route("GET /api/games")
    def _games(_):
        games = app.load_games()
        return {"count": len(games),
                "path": str(app.games_path) if app.games_path else None}

    @route("POST /api/finetune_all")
    def _finetune_all(_):
        games = app.load_games()
        if not games:
            raise ValueError("no saved games yet — play a few games first")
        report = app.bot.finetune_from_games(games)
        report_d = asdict(report)
        report_d["n_games"] = len(games)
        return report_d

    @route("POST /api/analyse")
    def _analyse(body):
        actions = parse_game(body if body else app.session.state())
        return app.bot.analyse_line(actions, top_k=int(body.get("top_k", 3)))

    @route("POST /api/finetune")
    def _finetune(body):
        if body.get("moves") or body.get("transcript") or body.get("history_actions"):
            actions = parse_game(body)
            human_color = body.get("human_color", app.session.state()["human_color"])
        else:
            st = app.session.state()
            actions = list(st["history_actions"])
            human_color = st["human_color"]
        report = app.bot.finetune_from_game(actions, human_color)
        return asdict(report)

    @route("POST /api/bot/reset")
    def _reset(_):
        app.bot.reset_to_baseline()
        return app.bot.info()

    # ---- request handler ------------------------------------------
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # keep the console quiet
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
                target = index  # SPA fallback for unknown client-side routes
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
            key = f"{method} {path}"
            handler = routes.get(key)
            if handler is None:
                self._send_json({"error": f"no route {key}"}, 404)
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
            try:
                with app.lock:
                    result = handler(body)
                self._send_json(result)
            except ValueError as e:
                self._send_json({"error": str(e)}, 400)
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                self._send_json({"error": f"{type(e).__name__}: {e}"}, 500)

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
          static_dir=None):
    app = AppState(bot, static_dir=static_dir)
    httpd = ThreadingHTTPServer((host, port), make_handler(app))
    return httpd

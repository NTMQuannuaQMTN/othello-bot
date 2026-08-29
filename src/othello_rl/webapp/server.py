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
import traceback
from dataclasses import asdict, is_dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict

from othello_rl.environment.board import Board
from .bot_service import OthelloBot
from .moves import parse_game, replay_positions
from .session import GameSession

STATIC_DIR = Path(__file__).parent / "static"
_CONTENT_TYPES = {".html": "text/html", ".js": "application/javascript",
                  ".css": "text/css", ".json": "application/json",
                  ".svg": "image/svg+xml", ".ico": "image/x-icon"}


class AppState:
    def __init__(self, bot: OthelloBot):
        self.bot = bot
        self.session = GameSession(bot)
        self.lock = threading.Lock()


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

    @route("POST /api/new")
    def _new(body):
        return app.session.new_game(body.get("human_color", "black"),
                                    int(body.get("level", 0)))

    @route("POST /api/move")
    def _move(body):
        return app.session.human_move(int(body["action"]))

    @route("POST /api/bot_move")
    def _bot_move(_):
        return app.session.bot_move()

    @route("POST /api/analyse")
    def _analyse(body):
        actions = parse_game(body if body else app.session.state())
        analyses = app.bot.analyse_game(actions, top_k=int(body.get("top_k", 3)))
        start_eval = app.bot.evaluate_position(Board.initial())["winprob_black"]
        graph = [{"ply": -1, "eval_black": start_eval}]
        graph += [{"ply": a.ply, "eval_black": a.eval_after_black} for a in analyses]
        return {
            "n_moves": len(actions),
            "actions": actions,
            "plies": [asdict(a) for a in analyses],
            "positions": replay_positions(actions),
            "eval_graph": graph,
            "summary": _summary(analyses),
        }

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

        def _send_file(self, path: Path):
            if not path.is_file() or STATIC_DIR not in path.resolve().parents:
                self._send_json({"error": "not found"}, 404)
                return
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", _CONTENT_TYPES.get(path.suffix, "application/octet-stream"))
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _dispatch(self, method: str):
            path = self.path.split("?", 1)[0]
            if method == "GET" and path in ("/", "/index.html"):
                self._send_file(STATIC_DIR / "index.html")
                return
            if method == "GET" and path.startswith("/static/"):
                self._send_file(STATIC_DIR / path[len("/static/"):])
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


def _summary(analyses) -> dict:
    counts: Dict[str, Dict[str, int]] = {"black": {}, "white": {}}
    for a in analyses:
        counts[a.side][a.label] = counts[a.side].get(a.label, 0) + 1
    return counts


def _json_default(o):
    if is_dataclass(o):
        return asdict(o)
    import numpy as np
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def serve(bot: OthelloBot, host: str = "127.0.0.1", port: int = 8000):
    app = AppState(bot)
    httpd = ThreadingHTTPServer((host, port), make_handler(app))
    return httpd

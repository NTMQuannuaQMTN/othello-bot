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

from othello_rl.environment.board import Board

from .bot_service import OthelloBot
from .moves import parse_game
from .session import GameSession


def _board_from(actions) -> Board:
    """Replay a flat action list (64 = pass) from the start position."""
    b = Board.initial()
    for a in (int(x) for x in actions):
        b = b.apply(None if (a == 64 or not b.legal_moves()) else divmod(a, 8))
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
        self.session = GameSession(bot)
        self.lock = threading.Lock()
        self.static_dir = Path(static_dir) if static_dir else DEFAULT_STATIC_DIR
        # the durable game dataset (append-only, deduplicated by move sequence)
        if games_path is not None:
            self.games_path = Path(games_path)
        elif bot.state_dir:
            self.games_path = Path(bot.state_dir) / "games.jsonl"
        else:
            self.games_path = None
        self._recorded = False  # has the *current* game already been appended?
        self._seen = set()      # move-sequence signatures already on disk
        if self.games_path and self.games_path.is_file():
            for g in self.load_games():
                self._seen.add(tuple(g.get("moves", [])))

    def _append_game(self, moves, human_color="black", learn_color=None) -> dict:
        """Append one game to the durable dataset, deduplicated by move sequence.
        Returns ``{saved: bool, count: int, reason?: str}``."""
        if not self.games_path:
            return {"saved": False, "count": 0, "reason": "no dataset configured"}
        moves = [int(a) for a in moves]
        if not moves:
            return {"saved": False, "count": len(self.load_games()), "reason": "empty game"}
        sig = tuple(moves)
        if sig in self._seen:
            return {"saved": False, "count": len(self.load_games()), "reason": "already saved"}
        # replay to record the result
        board = self.session.board.__class__.initial()
        for a in moves:
            mv = None if a == 64 or not board.legal_moves() else divmod(a, 8)
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

    def record_if_finished(self) -> None:
        """Auto-append the current session's game once it's over."""
        if not self.games_path or self._recorded:
            return
        st = self.session
        if not st.board.is_terminal() or len(st.history) == 0:
            return
        self._recorded = True
        self._append_game(list(st.history),
                          human_color="black" if st.human_color == 1 else "white")

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
    @route("GET /api/model")
    def _bot_info(_):
        d = dict(app.bot.info())
        d["dataset"] = str(app.games_path) if app.games_path else None
        d["dataset_games"] = len(app.load_games())
        return d

    @route("GET /api/state")
    def _state(_):
        return app.session.state()

    @route("GET /api/eval")
    @route("POST /api/eval")
    def _eval(body):
        """The eval bar (Black's look-ahead win probability). GET -> the live game
        position; POST ``{history_actions: [...]}`` -> the position after that
        prefix (so the bar tracks history navigation). Lightweight — one search,
        not one per legal move; use ``POST /api/analyse`` for the full picture."""
        acts = (body or {}).get("history_actions")
        b = app.session.board if acts is None else _board_from(acts)
        return app.bot.bar_eval(b)

    @route("POST /api/best_move")
    def _best_move(body):
        """The strongest move the engine can find for a position (alpha-beta +
        exact endgame). ``{history_actions: [...], time_budget?: 3.0}`` ->
        ``{action, san, winprob, winprob_stm, exact, depth, nodes, pv}``."""
        b = _board_from(body.get("history_actions") or
                        list(app.session.state()["history_actions"]))
        return app.bot.best_move(b, time_budget=float(body.get("time_budget", 3.0)),
                                 endgame_empties=int(body.get("endgame_empties", 16)))

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

    @route("POST /api/games")
    def _save_game(body):
        """Save an explicit game to the dataset (the Play tab also auto-saves).
        Body: ``{moves|transcript|history_actions, human_color?, learn_color?}``."""
        if body.get("moves") or body.get("transcript") or body.get("history_actions"):
            moves = parse_game(body)
        else:
            moves = list(app.session.state()["history_actions"])
        res = app._append_game(moves, human_color=body.get("human_color", "black"),
                               learn_color=body.get("learn_color"))
        res["path"] = str(app.games_path) if app.games_path else None
        return res

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
        """Fine-tune from a game. ``learn_color`` = which side's moves to learn
        from; defaults to the bot's colour in the current session (Play tab)."""
        if body.get("moves") or body.get("transcript") or body.get("history_actions"):
            actions = parse_game(body)
        else:
            actions = list(app.session.state()["history_actions"])
        learn_color = body.get("learn_color")
        if not learn_color:
            human = app.session.state()["human_color"]
            learn_color = "white" if human == "black" else "black"  # the bot's side
        report = app.bot.finetune_from_game(actions, learn_color)
        d = asdict(report)
        d["learn_color"] = learn_color
        return d

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
          static_dir=None, games_path=None):
    app = AppState(bot, static_dir=static_dir, games_path=games_path)
    httpd = ThreadingHTTPServer((host, port), make_handler(app))
    return httpd

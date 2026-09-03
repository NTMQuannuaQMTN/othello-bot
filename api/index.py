"""Vercel Python serverless entrypoint for the Othello bot API.

``vercel.json`` rewrites every ``/api/*`` request to this one function.  The bot
is loaded once per warm container from ``api/policy.npz`` (numpy only — no
PyTorch, which is what keeps this under the 250 MB function limit).  All API
logic lives in ``othello_rl.webapp.server``; this file is just the HTTP glue.
See ``docs/deploy.md``.
"""
import json
import sys
import traceback
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs

_HERE = Path(__file__).resolve().parent
for _p in (_HERE.parent / "src", _HERE.parent, _HERE):
    sp = str(_p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

# Load the bot once per container. If anything here fails, keep the failure so
# every request can report it as JSON instead of an opaque 500.
_boot_error = None
_bot = _app = _routes = None
_dispatch = _json_default = None
try:
    from othello_rl.webapp.bot_service import OthelloBot
    from othello_rl.webapp.server import (
        AppState, build_routes, dispatch as _dispatch, _json_default,
    )

    _policy = _HERE / "policy.npz"
    _bot = OthelloBot.load(str(_policy))
    _app = AppState(_bot, static_dir=None, games_path=None)
    _routes = build_routes(_app)
except Exception:  # noqa: BLE001
    _boot_error = traceback.format_exc()


def _api_path(raw: str, headers) -> str:
    """The real ``/api/...`` path this request was for — from ``self.path``, the
    ``?__vpath=`` query the rewrite adds, or an ``x-vercel-*`` header."""
    head, _, query = raw.partition("?")
    if head.startswith("/api/") and head not in ("/api/index", "/api"):
        return head
    if query:
        v = parse_qs(query).get("__vpath", [""])[0]
        if v:
            return v if v.startswith("/api/") else "/api/" + v.lstrip("/")
    for h in ("x-vercel-original-pathname", "x-forwarded-uri", "x-original-uri"):
        v = headers.get(h)
        if v and v.split("?", 1)[0].startswith("/api/"):
            return v.split("?", 1)[0]
    return head if head.startswith("/api/") else "/api/" + head.lstrip("/")


def _plain_default(o):
    return _json_default(o) if _json_default else str(o)


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self._run("GET")

    def do_POST(self):
        self._run("POST")

    def do_OPTIONS(self):
        self._send(204, None)

    def log_message(self, *a):
        pass

    def _run(self, method: str):
        path = _api_path(self.path, self.headers)
        if _boot_error is not None:
            return self._send(500, {"error": "bot failed to load",
                                    "detail": _boot_error,
                                    "python": sys.version,
                                    "sys_path": sys.path[:6]})
        body = {}
        if method == "POST":
            n = int(self.headers.get("content-length", 0) or 0)
            raw = self.rfile.read(n) if n else b""
            if raw:
                try:
                    body = json.loads(raw)
                except json.JSONDecodeError:
                    return self._send(400, {"error": "invalid JSON body"})
        try:
            status, result = _dispatch(_app, _routes, method, path, body)
        except Exception:  # noqa: BLE001
            return self._send(500, {"error": "request failed", "path": path,
                                    "detail": traceback.format_exc()})
        self._send(status, result)

    def _send(self, status: int, obj):
        data = b"" if obj is None else json.dumps(obj, default=_plain_default).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if data:
            self.wfile.write(data)

"""Vercel Python serverless entrypoint for the Othello bot API.

Catch-all: Vercel routes every ``/api/*`` request to this function.  The bot is
loaded once per warm container from ``api/policy.npz`` (numpy only — no PyTorch,
which is what keeps this under the 250 MB function limit).  All API logic lives
in ``othello_rl.webapp.server``; this file is just the HTTP glue.
See ``docs/deploy.md``.
"""
import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

from othello_rl.webapp.bot_service import OthelloBot          # noqa: E402
from othello_rl.webapp.server import (                        # noqa: E402
    AppState, build_routes, dispatch, _json_default,
)

_bot = OthelloBot.load(str(_HERE / "policy.npz"))
_app = AppState(_bot, static_dir=None, games_path=None)       # read-only deploy
_routes = build_routes(_app)


def _api_path(raw: str, headers) -> str:
    path = raw.split("?", 1)[0]
    if path.startswith("/api/"):
        return path
    # defensive: some routing configs hand the function a stripped path
    for h in ("x-vercel-original-pathname", "x-forwarded-uri", "x-original-uri"):
        v = headers.get(h)
        if v and v.split("?", 1)[0].startswith("/api/"):
            return v.split("?", 1)[0]
    return "/api/" + path.lstrip("/")


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
        body = {}
        if method == "POST":
            n = int(self.headers.get("content-length", 0) or 0)
            raw = self.rfile.read(n) if n else b""
            if raw:
                try:
                    body = json.loads(raw)
                except json.JSONDecodeError:
                    return self._send(400, {"error": "invalid JSON body"})
        status, result = dispatch(_app, _routes, method, path, body)
        self._send(status, result)

    def _send(self, status: int, obj):
        data = b"" if obj is None else json.dumps(obj, default=_json_default).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if data:
            self.wfile.write(data)

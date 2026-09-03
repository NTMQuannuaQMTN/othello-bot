import json
import threading
import urllib.error
import urllib.request

import pytest

from othello_rl.environment.board import Board
from othello_rl.rl.agent import DQNAgent, NetworkConfig
from othello_rl.webapp.bot_service import FineTuneConfig, OthelloBot
from othello_rl.webapp.moves import parse_game, replay_positions
from othello_rl.webapp.server import serve


# --------------------------- moves parsing --------------------------- #
def test_parse_game_from_action_list():
    # round-trip: transcript -> actions -> same actions
    acts = parse_game("f5d6c3")
    assert parse_game(acts) == acts
    assert acts[0] == 4 * 8 + 5  # f5


def test_parse_game_from_transcript_variants():
    a1 = parse_game("f5d6c3d3c4")
    a2 = parse_game("f5 d6 c3 d3 c4")
    a3 = parse_game({"moves": ["f5", "d6", "c3", "d3", "c4"]})
    assert a1 == a2 == a3
    assert len(a1) == 5


def test_parse_game_rejects_illegal():
    with pytest.raises(ValueError):
        parse_game(["a1"])  # not a legal opening move


def test_replay_positions_length_and_content():
    acts = parse_game("f5d6c3")
    pos = replay_positions(acts)
    assert len(pos) == len(acts) + 1
    assert pos[0]["turn"] == "black"
    assert sum(1 for row in pos[0]["grid"] for v in row if v != 0) == 4


# --------------------------- HTTP API --------------------------- #
@pytest.fixture(scope="module")
def server():
    agent = DQNAgent(NetworkConfig(channels=8, blocks=2, hidden=16), seed=0)
    bot = OthelloBot(agent, ft_config=FineTuneConfig(
        grad_steps=6, batch_size=16, anchor_transitions=120, guardrail_games=4,
        buffer_capacity=1500, grade_lookahead=2))
    httpd = serve(bot, port=8912)
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    yield "http://127.0.0.1:8912"
    httpd.shutdown()


def _call(base, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


class _Client:
    """The API is stateless — the client carries its own move history."""

    def __init__(self, base, human_color="black"):
        self.base, self.hc, self.hist = base, human_color, []

    def _track(self, st):
        self.hist = list(st["history_actions"])
        return st

    def new(self):
        return self._track(_call(self.base, "/api/new", {"human_color": self.hc}))

    def move(self, action, bot_reply=True):
        return self._track(_call(self.base, "/api/move", {
            "human_color": self.hc, "history_actions": self.hist,
            "action": action, "bot_reply": bot_reply}))

    def bot_move(self):
        return self._track(_call(self.base, "/api/bot_move", {
            "human_color": self.hc, "history_actions": self.hist}))


def test_move_bot_reply_false_defers_the_bot(server):
    c = _Client(server, "black")
    st = c.new()
    after = c.move(st["legal_actions"][0], bot_reply=False)
    assert after["ply"] == 1                 # only the human move applied
    assert after["turn"] == "white" and not after["your_turn"]
    assert after["last_bot_moves"] == []
    st2 = c.bot_move()                       # the client then triggers the reply
    assert st2["ply"] == 2 and st2["your_turn"]
    assert len(st2["last_bot_moves"]) == 1


def test_bot_move_tolerates_a_get(server):
    with urllib.request.urlopen(server + "/api/bot_move") as r:  # GET, no body/history
        assert r.status == 200
        assert "grid" in json.loads(r.read())


def test_games_are_recorded_and_finetune_all(tmp_path):
    from othello_rl.webapp.server import serve as _serve
    agent = DQNAgent(NetworkConfig(channels=8, blocks=2, hidden=16), seed=1)
    bot = OthelloBot(agent, state_dir=str(tmp_path), ft_config=FineTuneConfig(
        grad_steps=4, batch_size=16, anchor_transitions=80, guardrail_games=4,
        buffer_capacity=1000, grade_lookahead=2))
    httpd = _serve(bot, port=8913)
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    base = "http://127.0.0.1:8913"
    try:
        import random
        assert _call(base, "/api/games")["count"] == 0
        for g in range(2):
            c = _Client(base, "black" if g == 0 else "white")
            st = c.new()
            rng = random.Random(g)
            while not st["game_over"]:
                st = (c.move(rng.choice(st["legal_actions"]))
                      if st["your_turn"] else c.bot_move())
        games = _call(base, "/api/games")
        assert games["count"] == 2
        recs = [json.loads(l) for l in (tmp_path / "games.jsonl").read_text().splitlines()]
        assert len(recs) == 2
        assert recs[0]["human_color"] == "black" and recs[1]["human_color"] == "white"
        assert all(r["winner"] in ("black", "white", "draw") and "moves" in r for r in recs)

        rep = _call(base, "/api/finetune_all", {})
        assert rep["n_games"] == 2 and "loss_after" in rep
    finally:
        httpd.shutdown()


def test_explicit_save_game_and_dedup(tmp_path):
    from othello_rl.webapp.server import serve as _serve
    agent = DQNAgent(NetworkConfig(channels=8, blocks=2, hidden=16), seed=2)
    bot = OthelloBot(agent, state_dir=str(tmp_path))
    httpd = _serve(bot, port=8914)
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    base = "http://127.0.0.1:8914"
    try:
        r1 = _call(base, "/api/games", {"transcript": "f5d6c3d3c4f4"})
        assert r1["saved"] is True and r1["count"] == 1
        r2 = _call(base, "/api/games", {"transcript": "f5d6c3d3c4f4"})   # same line
        assert r2["saved"] is False and r2["reason"] == "already saved"
        assert _call(base, "/api/games")["count"] == 1
        rec = json.loads((tmp_path / "games.jsonl").read_text().splitlines()[0])
        assert rec["winner"] in ("black", "white", "draw") and len(rec["moves"]) == 6
    finally:
        httpd.shutdown()


def test_index_serves_html(server):
    # serves web/dist/index.html when the React app is built, otherwise a
    # fallback info page — either way it's HTML with a mounting point.
    with urllib.request.urlopen(server + "/") as r:
        body = r.read()
    assert body.lstrip().lower().startswith(b"<!doctype html")
    assert b'id="root"' in body or b"npm run" in body


def test_unknown_path_spa_fallback(server):
    # client-side routes fall back to index.html (or the info page)
    with urllib.request.urlopen(server + "/some/deep/route") as r:
        assert r.status == 200
        assert r.read().lstrip().lower().startswith(b"<!doctype html")


def test_play_and_analyse_flow(server):
    info = _call(server, "/api/bot")
    assert info["version"] == 0 and info["params"] > 0

    c = _Client(server, "black")
    st = c.new()
    assert st["your_turn"] and len(st["legal_actions"]) == 4
    assert st["moves"] == []  # no moves yet

    import random
    rng = random.Random(3)
    while not st["game_over"]:
        st = c.move(rng.choice(st["legal_actions"])) if st["your_turn"] else c.bot_move()
    assert st["winner"] in ("black", "white", "draw")

    # per-move log: numbered, correct side (accounts for passes), who played it
    log = st["moves"]
    assert len(log) == st["ply"]
    assert [m["n"] for m in log] == list(range(1, len(log) + 1))
    assert log[0]["side"] == "black" and log[0]["by"] == "you"  # human is black, moves first
    for m in log:
        assert m["side"] in ("black", "white") and m["by"] in ("you", "bot")
    # a bot move is by the bot and by white (human is black)
    bot_moves = [m for m in log if m["by"] == "bot"]
    assert bot_moves and all(m["side"] == "white" for m in bot_moves)

    # per-ply board grids for history navigation in the Play tab
    pos = st["positions"]
    assert len(pos) == st["ply"] + 1
    assert pos[0][3][3] == -1 and pos[0][3][4] == 1  # initial position
    assert pos[-1] == st["grid"]                     # last == current
    assert all(len(g) == 8 and len(g[0]) == 8 for g in pos)

    an = _call(server, "/api/analyse", {"history_actions": st["history_actions"]})
    assert len(an["eval_graph"]) == an["n_moves"] + 1
    assert len(an["positions"]) == an["n_moves"] + 1
    assert all(0.0 <= p["eval_black"] <= 1.0 for p in an["eval_graph"])
    # each position is navigation-ready (grid + legal moves + eval)
    assert all("eval" in p and "legal_actions" in p for p in an["positions"])

    # interactive: analyse a partial line and get the current position's options
    partial = _call(server, "/api/analyse", {"moves": st["history_actions"][:5]})
    cur = partial["positions"][-1]
    assert cur["legal_actions"] and cur["eval"]["moves"]

    # eval bar tracks history navigation: GET = live, POST prefix = that position
    live = _call(server, "/api/eval")
    hist = _call(server, "/api/eval", {"history_actions": st["history_actions"][:6]})
    assert 0.0 <= hist["winprob_black"] <= 1.0
    assert hist != live or st["ply"] == 6


def test_illegal_move_returns_400(server):
    _call(server, "/api/new", {"human_color": "black"})
    try:
        _call(server, "/api/move", {"action": 0})
        assert False, "expected HTTP 400"
    except urllib.error.HTTPError as e:
        assert e.code == 400
        assert "illegal" in json.loads(e.read())["error"].lower()


def test_finetune_endpoint(server):
    c = _Client(server, "white")
    st = c.new()
    import random
    rng = random.Random(5)
    while not st["game_over"]:
        st = c.move(rng.choice(st["legal_actions"])) if st["your_turn"] else c.bot_move()
    rep = _call(server, "/api/finetune",
                {"history_actions": st["history_actions"], "learn_color": "black"})
    assert "grades" in rep and "loss_after" in rep
    assert isinstance(rep["rolled_back"], bool)

    reset = _call(server, "/api/bot/reset", {})
    assert reset["version"] == 0

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
        buffer_capacity=1500))
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

    st = _call(server, "/api/new", {"human_color": "black"})
    assert st["your_turn"] and len(st["legal_actions"]) == 4

    import random
    rng = random.Random(3)
    while not st["game_over"]:
        if st["your_turn"]:
            st = _call(server, "/api/move", {"action": rng.choice(st["legal_actions"])})
        else:
            st = _call(server, "/api/bot_move")
    assert st["winner"] in ("black", "white", "draw")

    an = _call(server, "/api/analyse", {"history_actions": st["history_actions"]})
    assert an["n_moves"] == len(st["history_actions"])
    assert len(an["eval_graph"]) == an["n_moves"] + 1
    assert len(an["positions"]) == an["n_moves"] + 1
    assert all(0.0 <= p["eval_black"] <= 1.0 for p in an["eval_graph"])


def test_illegal_move_returns_400(server):
    _call(server, "/api/new", {"human_color": "black"})
    try:
        _call(server, "/api/move", {"action": 0})
        assert False, "expected HTTP 400"
    except urllib.error.HTTPError as e:
        assert e.code == 400
        assert "illegal" in json.loads(e.read())["error"].lower()


def test_finetune_endpoint(server):
    st = _call(server, "/api/new", {"human_color": "white"})
    import random
    rng = random.Random(5)
    while not st["game_over"]:
        st = (_call(server, "/api/move", {"action": rng.choice(st["legal_actions"])})
              if st["your_turn"] else _call(server, "/api/bot_move"))
    rep = _call(server, "/api/finetune", {})
    assert "grades" in rep and "loss_after" in rep
    assert isinstance(rep["rolled_back"], bool)

    reset = _call(server, "/api/bot/reset", {})
    assert reset["version"] == 0

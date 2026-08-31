"""Tests for the Egaroucid GTP bridge (``othello_rl.eval_external``).

None of these need the Egaroucid executable: the game-loop tests drive
``play_game`` against a **fake engine** that reproduces Egaroucid's documented
GTP semantics (auto-pass when told to move for the "wrong" colour, ``genmove``
returns ``PASS`` without touching the board).  One optional test runs a real
game if a built executable happens to be on disk.
"""
from __future__ import annotations

import io

import pytest

from othello_rl.agents import GreedyAgent, RandomAgent
from othello_rl.environment.board import (
    BLACK,
    WHITE,
    Board,
    action_to_rc,
    opponent,
    square_name,
)
from othello_rl.environment.environment import legal_action_mask
from othello_rl.eval_external.egaroucid import (
    EgaroucidEngine,
    EgaroucidError,
    coord_to_gtp,
    find_egaroucid,
    gtp_to_coord,
)
from othello_rl.eval_external.match import GameRecord, play_game, run_match


# --------------------------------------------------------------------------- #
# coordinate conversions
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("gtp,rc", [
    ("A1", (0, 0)), ("H1", (0, 7)), ("A8", (7, 0)), ("H8", (7, 7)),
    ("D3", (2, 3)), ("C4", (3, 2)), ("F5", (4, 5)),
])
def test_coordinate_roundtrip(gtp, rc):
    assert gtp_to_coord(gtp) == rc
    assert coord_to_gtp(rc).upper() == gtp
    # and it matches the project's own square naming (GTP is just upper-case)
    assert square_name(rc) == gtp.lower()


def test_pass_coordinate():
    assert gtp_to_coord("pass") is None
    assert gtp_to_coord("PASS") is None
    assert coord_to_gtp(None) == "pass"


def test_bad_coordinates_raise():
    with pytest.raises(ValueError):
        gtp_to_coord("Z9")
    with pytest.raises(ValueError):
        coord_to_gtp((8, 0))


def test_find_egaroucid_missing_is_informative():
    with pytest.raises(EgaroucidError) as ei:
        find_egaroucid("/no/such/egaroucid/binary")
    assert "not found" in str(ei.value)


# --------------------------------------------------------------------------- #
# GTP response parser (fake subprocess)
# --------------------------------------------------------------------------- #
class _FakeProc:
    def __init__(self, script):
        self._out = io.StringIO(script)
        self.stdin = io.StringIO()
        self.stdout = self._out
        self.returncode = None

    def poll(self):
        return None


def _engine_with_script(script):
    eng = EgaroucidEngine.__new__(EgaroucidEngine)
    eng.proc = _FakeProc(script)
    eng.move_timeout = 5.0
    return eng


def test_send_parses_simple_and_error_replies():
    eng = _engine_with_script("= D3\n\n= F5\n\n? illegal move\n\n")
    assert eng.send("genmove black") == "D3"
    assert eng.send("genmove white") == "F5"
    with pytest.raises(EgaroucidError):
        eng.send("play black z9")


def test_send_parses_multiline_reply():
    eng = _engine_with_script("= quit\nprotocol_version\nplay\ngenmove\n\n")
    body = eng.send("list_commands")
    assert body.splitlines()[0] == "quit"
    assert "genmove" in body


# --------------------------------------------------------------------------- #
# a fake engine that mirrors Egaroucid's GTP semantics
# --------------------------------------------------------------------------- #
class FakeEngine:
    """Plays via the project rules; reproduces Egaroucid's GTP quirks."""

    name = "FakeEngine"
    version = "0.0-test"
    protocol_version = "2.0"

    def __init__(self, opponent_agent=None, seed=0):
        self._agent = opponent_agent or GreedyAgent()
        self._seed = seed
        self.board = Board.initial()
        self.calls = []

    # -- GTP surface -------------------------------------------------
    def clear_board(self):
        self.board = Board.initial()
        self.calls.append("clear_board")

    def _autopass_to(self, color):
        want = BLACK if color == "black" else WHITE
        if self.board.player != want and not self.board.is_terminal():
            # Egaroucid: play/genmove for the "wrong" colour passes internally
            self.board = self.board.apply(None)

    def play(self, color, move):
        self.calls.append(("play", color, coord_to_gtp(move)))
        self._autopass_to(color)
        self.board = self.board.apply(move)

    def genmove(self, color):
        self.calls.append(("genmove", color))
        self._autopass_to(color)
        if not self.board.legal_moves():
            return None                      # Egaroucid prints PASS, board untouched
        mv = self._agent.select_move(self.board)
        self.board = self.board.apply(mv)
        return mv

    def final_score(self):
        b, w = self.board.scores()
        if b > w:
            return f"B{b - w}"
        if w > b:
            return f"W{b - w}"
        return "0"

    def final_result(self):
        b, w = self.board.scores()
        if b == w:
            return "Draw."
        side = "Black" if b > w else "White"
        return f"{side} wins by {abs(b - w)} points. Final score is B {b} and W {w}"

    def describe(self):
        return {"name": self.name, "version": self.version,
                "protocol_version": self.protocol_version, "executable": "<fake>",
                "level": 0, "threads": 1, "argv": []}

    def close(self):
        self.calls.append("close")


# --------------------------------------------------------------------------- #
# game-loop behaviour
# --------------------------------------------------------------------------- #
def _bot():
    from othello_rl.rl.checkpoint import Registry, load_agent

    class _BotAdapter:
        def __init__(self, agent):
            self.agent = agent

        def select_action(self, board):
            from othello_rl.environment.board import PASS_ACTION
            from othello_rl.environment.environment import encode_observation, legal_action_mask
            if not board.legal_moves():
                return PASS_ACTION
            return int(self.agent.greedy_act(encode_observation(board),
                                             legal_action_mask(board)))

    return _BotAdapter(load_agent(Registry.load().active_checkpoint_path()))


@pytest.mark.parametrize("rl_is_black", [True, False])
def test_play_game_completes_and_scores_consistently(rl_is_black):
    rec = play_game(_bot(), FakeEngine(GreedyAgent()), rl_is_black=rl_is_black,
                    opening_plies=4, verbose=False)
    assert isinstance(rec, GameRecord)
    assert rec.error is None
    assert rec.score_black + rec.score_white <= 64
    assert rec.n_moves == len(rec.moves)
    # winner / result agree with the disc counts
    if rec.score_black > rec.score_white:
        assert rec.winner == "black"
    elif rec.score_white > rec.score_black:
        assert rec.winner == "white"
    else:
        assert rec.winner == "draw"
    rl_win = (rec.winner == rec.rl_color)
    assert (rec.result == "rl_win") == rl_win or rec.winner == "draw"

    # the transcript (real placements only) replays cleanly to the same result;
    # Board.apply performs the forced passes implicitly.
    from othello_rl.environment.board import parse_square
    replay = Board.initial()
    auto_skips = 0
    for san in rec.transcript.split():
        mv = parse_square(san)
        assert mv in replay.legal_moves()
        mover = replay.player
        replay = replay.apply(mv)
        if not replay.is_terminal() and replay.player == mover:
            auto_skips += 1
    assert replay.is_terminal()
    assert replay.scores() == (rec.score_black, rec.score_white)
    assert rec.egaroucid_agrees is True
    # every logged pass corresponds to a real forced skip (± a trailing pass)
    assert auto_skips <= rec.n_passes <= auto_skips + 2
    # the pass annotations and the real moves together number n_moves
    assert sum(not m["pass"] for m in rec.moves) == len(rec.transcript.split())


def test_forced_passes_are_logged_and_counted():
    """A near-full board where one side is about to run out of moves — exercises
    the auto-skip -> explicit-pass path."""
    rec = play_game(_bot(), FakeEngine(RandomAgent(seed=3)), rl_is_black=True,
                    opening_plies=0, verbose=False)
    logged_passes = sum(1 for m in rec.moves if m["pass"])
    assert logged_passes == rec.n_passes
    # each pass annotation sits right after a move by the *other* colour
    for i, m in enumerate(rec.moves):
        if m["pass"]:
            assert i > 0 and rec.moves[i - 1]["player"] != m["player"]
    # and the real-move transcript still replays to the recorded score
    from othello_rl.environment.board import parse_square
    replay = Board.initial()
    for san in rec.transcript.split():
        replay = replay.apply(parse_square(san))
    assert replay.is_terminal() and replay.scores() == (rec.score_black, rec.score_white)


def test_illegal_rl_move_stops_the_game():
    class BadBot:
        agent = None

        def select_action(self, board):
            # a square that is never legal from the opening
            return 0

    from othello_rl.eval_external.match import IllegalRLMove
    with pytest.raises(IllegalRLMove):
        play_game(BadBot(), FakeEngine(), rl_is_black=True, opening_plies=0,
                  verbose=False)


def test_run_match_alternates_colours_and_totals_add_up():
    summary = run_match(_bot(), FakeEngine(GreedyAgent()), games=4,
                        opening_plies=4, seed=1, verbose=False)
    assert summary.games == 4
    assert summary.rl_wins + summary.egaroucid_wins + summary.draws == 4
    assert summary.rl_black_games == 2 and summary.rl_white_games == 2
    assert [r.rl_color for r in summary.records] == ["black", "white", "black", "white"]
    assert 0.0 <= summary.win_rate <= 1.0


# --------------------------------------------------------------------------- #
# optional: a real game against a built Egaroucid, if one is on disk
# --------------------------------------------------------------------------- #
def test_real_egaroucid_one_game_if_available():
    try:
        exe = find_egaroucid()
    except EgaroucidError:
        pytest.skip("Egaroucid executable not built/available")
    with EgaroucidEngine(str(exe), level=1, threads=1) as eng:
        rec = play_game(_bot(), eng, rl_is_black=True, opening_plies=4, verbose=False)
    assert rec.error is None
    assert rec.score_black + rec.score_white <= 64
    assert rec.n_moves > 10
    assert rec.egaroucid_agrees is True
    from othello_rl.environment.board import parse_square
    replay = Board.initial()
    for san in rec.transcript.split():
        replay = replay.apply(parse_square(san))
    assert replay.is_terminal() and replay.scores() == (rec.score_black, rec.score_white)

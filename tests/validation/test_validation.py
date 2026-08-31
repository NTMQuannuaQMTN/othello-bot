import json
import random

from othello_rl.environment.board import Board, PASS_ACTION
from othello_rl.ingest.records import GameRecord
from othello_rl.validation import Status, validate, validate_file


def _random_game(seed):
    """Play a full random game -> (placement actions, winner). Board.apply
    auto-skips forced passes, so placements never contain PASS_ACTION."""
    rng = random.Random(seed)
    s = Board.initial()
    placements = []
    while not s.is_terminal():
        r, c = rng.choice(s.legal_moves())
        placements.append(r * 8 + c)
        s = s.apply((r, c))
    w = s.winner()
    return placements, ("black" if w == 1 else "white" if w == -1 else "draw")


def _rec(moves, **kw):
    return GameRecord(source="t", source_format="t", moves=list(moves), **kw)


def test_valid_full_game_reconstructs():
    placements, winner = _random_game(3)
    res = validate(_rec(placements, result={"winner": winner}))
    assert res.status is Status.VALID
    assert res.canonical_moves == placements
    assert res.final_black + res.final_white <= 64
    assert res.replayed_winner == winner and res.winner_matches is True


def test_explicit_pass_tokens_are_dropped_not_rejected():
    # WThor omits forced passes; some sources annotate them. Either way, the
    # engine auto-handles passes, so an explicit 64 is dropped, not an error.
    placements, winner = _random_game(6)
    annotated = [placements[0], PASS_ACTION, placements[1], PASS_ACTION, *placements[2:]]
    res = validate(_rec(annotated, result={"winner": winner}))
    assert res.status is Status.VALID
    assert res.canonical_moves == placements and res.passes_skipped == 2


def test_illegal_move_is_invalid_at_the_right_ply():
    placements, _ = _random_game(1)
    bad = placements[:6] + [placements[0]]          # replay the first move again
    res = validate(_rec(bad))
    assert res.status is Status.INVALID
    assert "illegal move" in res.reason and "ply 6" in res.reason


def test_truncated_game_is_incomplete():
    placements, _ = _random_game(2)
    res = validate(_rec(placements[:10]))
    assert res.status is Status.INCOMPLETE


def test_moves_after_game_end_is_invalid():
    placements, _ = _random_game(4)
    res = validate(_rec(placements + [0, 1, 2]))
    assert res.status is Status.INVALID and "after the game ended" in res.reason


def test_empty_game_is_unsupported():
    assert validate(_rec([])).status is Status.UNSUPPORTED_FORMAT
    assert validate(_rec([PASS_ACTION, PASS_ACTION])).status is Status.UNSUPPORTED_FORMAT


def test_validation_is_deterministic():
    placements, _ = _random_game(7)
    a, b = validate(_rec(placements)), validate(_rec(placements))
    assert a.status == b.status and a.canonical_moves == b.canonical_moves
    assert (a.final_black, a.final_white) == (b.final_black, b.final_white)


def test_winner_mismatch_is_kept_but_flagged():
    placements, winner = _random_game(8)
    wrong = "white" if winner != "white" else "black"
    res = validate(_rec(placements, result={"winner": wrong}))
    assert res.status is Status.VALID and res.winner_matches is False


def test_pipeline_routes_valid_and_invalid(tmp_path):
    placements, winner = _random_game(9)
    good = _rec(placements, game_id="g_good", result={"winner": winner})
    bad = _rec(placements[:5] + [placements[0]], game_id="g_bad")
    src = tmp_path / "in.jsonl"
    src.write_text(good.to_json() + "\n" + bad.to_json() + "\n")
    valid_out, rej_out = tmp_path / "valid.jsonl", tmp_path / "rejected.jsonl"
    stats = validate_file(src, valid_out, rej_out, source="t", report_dir=tmp_path / "rep")

    assert stats.valid == 1 and stats.invalid == 1
    vrecs = [json.loads(l) for l in valid_out.read_text().splitlines()]
    assert [r["game_id"] for r in vrecs] == ["g_good"]
    assert vrecs[0]["canonical_moves"] == placements
    rrecs = [json.loads(l) for l in rej_out.read_text().splitlines()]
    assert rrecs[0]["game_id"] == "g_bad" and rrecs[0]["status"] == "INVALID"
    assert (tmp_path / "rep" / "validation.stats.json").is_file()
    assert json.loads(valid_out.with_suffix(".stats.json").read_text())["valid"] == 1

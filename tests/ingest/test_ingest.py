import json
import struct

import pytest

from othello_rl.ingest import Deduplicator, GameRecord, ingest, read_records
from othello_rl.ingest.sources import UnsupportedFormat, get_source


# --------------------------------------------------------------------------- #
# WThor .wtb
# --------------------------------------------------------------------------- #
def _wtb_bytes(games):
    """games: list of (trn, black_no, white_no, real_score, [move_bytes])"""
    hdr = bytes([20, 26, 1, 1]) + struct.pack("<I", len(games)) + \
        struct.pack("<H", 0) + struct.pack("<H", 2026) + bytes([8, 0, 22, 0])
    body = b""
    for trn, b, w, score, moves in games:
        rec = struct.pack("<HHH", trn, b, w) + bytes([score, 0])
        mv = bytes(moves) + bytes(60 - len(moves))
        body += rec + mv
    return hdr + body


def test_wthor_parses_moves_metadata_and_result(tmp_path):
    # d3, c3, c4  ->  bytes 34, 33, 43  ->  actions 19, 18, 26
    wtb = tmp_path / "TEST.wtb"
    wtb.write_bytes(_wtb_bytes([(5, 10, 20, 40, [34, 33, 43])]))
    recs = list(get_source("wthor").parse(wtb))
    assert len(recs) == 1
    r = recs[0]
    assert r.moves == [19, 18, 26]
    assert r.source == "wthor" and r.source_format == "wtb"
    assert r.result == {"black_discs": 40, "white_discs": 24, "winner": "black"}
    assert r.metadata["year"] == 2026 and r.metadata["tournament_no"] == 5
    assert r.provenance["pass_convention"] == "implicit"


def test_wthor_skips_a_corrupt_game_keeps_the_rest(tmp_path):
    wtb = tmp_path / "MIX.wtb"
    wtb.write_bytes(_wtb_bytes([
        (1, 1, 1, 32, [99, 34]),      # 99 -> row 9: bad, whole game skipped
        (2, 2, 2, 40, [34, 43]),      # fine
    ]))
    recs = list(get_source("wthor").parse(wtb))
    assert [r.moves for r in recs] == [[19, 26]]


def test_wthor_rejects_non_wtb(tmp_path):
    p = tmp_path / "x.wtb"
    p.write_bytes(b"nope")
    with pytest.raises(UnsupportedFormat):
        list(get_source("wthor").parse(p))


# --------------------------------------------------------------------------- #
# transcript / jsonl / generic
# --------------------------------------------------------------------------- #
def test_transcript_source(tmp_path):
    p = tmp_path / "g.txt"
    p.write_text("# a comment\nf5 d6 c3 d3 c4\nf5d6c3\n")
    recs = list(get_source("transcript").parse(p))
    assert len(recs) == 2
    assert recs[0].moves[0] == 4 * 8 + 5  # f5


def test_jsonl_source_reads_our_webapp_format(tmp_path):
    p = tmp_path / "games.jsonl"
    p.write_text(json.dumps({"moves": [19, 18, 26], "winner": "white",
                             "score": {"black": 20, "white": 44}}) + "\n")
    r = list(get_source("jsonl").parse(p))[0]
    assert r.moves == [19, 18, 26]
    assert r.result["winner"] == "white" and r.result["white_discs"] == 44
    assert r.provenance["pass_convention"] == "explicit"


def test_generic_json_source(tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps({"games": [{"moves": ["f5", "d6", "pass", "c3"]}]}))
    r = list(get_source("generic").parse(p))[0]
    assert r.moves == [37, 43, 64, 18]


def test_unknown_source_raises():
    with pytest.raises(UnsupportedFormat):
        get_source("nope")


# --------------------------------------------------------------------------- #
# record + dedup + pipeline
# --------------------------------------------------------------------------- #
def test_gamerecord_json_roundtrip():
    r = GameRecord(source="s", source_format="f", moves=[19, 64, 26],
                   metadata={"x": 1}, result={"winner": "black"})
    r2 = GameRecord.from_json(r.to_json())
    assert r2.moves == r.moves and r2.metadata == r.metadata and r2.game_id == r.game_id
    # dedup key ignores passes
    assert r.move_signature() == "19,26"


def test_gamerecord_rejects_bad_data_kind():
    with pytest.raises(ValueError):
        GameRecord(source="s", source_format="f", moves=[1], data_kind="bogus")


def test_deduplicator():
    dd = Deduplicator()
    a = GameRecord(source="a", source_format="f", moves=[19, 26])
    b = GameRecord(source="b", source_format="f", moves=[19, 64, 26])  # same, +pass
    c = GameRecord(source="a", source_format="f", moves=[19, 18])
    assert dd.is_new(a) and not dd.is_new(b) and dd.is_new(c)
    assert dd.stats() == {"kept": 2, "duplicates": 1, "unique": 2}


def test_ingest_pipeline_writes_jsonl_and_stats(tmp_path):
    src = tmp_path / "TEST.wtb"
    src.write_bytes(_wtb_bytes([(1, 1, 1, 40, [34, 43]),
                                (2, 2, 2, 30, [34, 43]),   # duplicate moves
                                (3, 3, 3, 20, [34, 33])]))
    out = tmp_path / "out" / "wthor.raw.jsonl"
    stats = ingest("wthor", src, out, dedup=True)
    assert stats.parsed == 3 and stats.duplicates == 1 and stats.written == 2
    recs = read_records(out)
    assert len(recs) == 2
    assert json.loads(out.with_suffix(".ingest.json").read_text())["written"] == 2

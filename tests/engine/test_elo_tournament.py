"""Smoke-test the Elo tournament (agent-only, no Egaroucid)."""
import importlib.util
import json
from pathlib import Path


def _mod():
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "elo_tournament", root / "scripts" / "elo_tournament.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_agent_only_tournament(tmp_path):
    mod = _mod()
    rc = mod.main(["--games", "2", "--agent-games", "4", "--minimax-depths", "1,2",
                   "--eg-levels", "0", "--eg-opponents", "", "--eg-vs-eg", "0",
                   "--engine-budget", "0.03", "--engine-endgame", "6",
                   "--out", str(tmp_path)])
    assert rc == 0
    t = json.loads(next(tmp_path.glob("tournament_*.json")).read_text())
    r = t["ratings"]
    assert {"engine", "random", "greedy", "heuristic", "minimax:1", "minimax:2"} <= set(r)
    # sane ordering: everything beats random, engine/minimax beat greedy
    assert r["random"] == min(r.values())
    assert r["engine"] > r["greedy"] and r["minimax:2"] > r["greedy"]
    anc = json.loads((tmp_path / "egaroucid_anchors.json").read_text())
    assert "our_bot_elo" in anc and "egaroucid_elo" in anc

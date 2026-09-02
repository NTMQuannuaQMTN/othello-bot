"""Smoke-test the engine-distillation trainer (one tiny round)."""
import importlib.util
import json
from pathlib import Path


def _load():
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "train_from_engine", root / "scripts" / "train_from_engine.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_generate_game_labels_are_legal_engine_moves():
    import random
    from othello_rl.environment.board import Board, action_to_rc
    mod = _load()
    g = mod.generate_game(random.Random(0), budget=0.01, endgame=6,
                          opening_plies=2, explore=0.3)
    assert len(g["moves"]) == len(g["labels"]) >= 10
    assert g["winner"] in ("black", "white", "draw")
    # every label was legal at its position
    b = Board.initial()
    for played, label in zip(g["moves"], g["labels"]):
        while not b.legal_moves() and not b.is_terminal():
            b = b.apply(None)
        if b.is_terminal():
            break
        assert action_to_rc(label) in b.legal_moves()
        b = b.apply(action_to_rc(played))
    ex = list(mod.game_to_examples(g))
    assert ex and ex[0][0].shape == (3, 8, 8) and 0 <= ex[0][2] < 64


def test_trainer_runs_one_round(tmp_path):
    mod = _load()
    out = tmp_path / "run"
    rc = mod.main(["--seconds", "1", "--engine-budget", "0.01", "--engine-endgame", "6",
                   "--games-per-round", "3", "--grad-steps", "10", "--batch-size", "32",
                   "--eval-games", "6", "--out", str(out)])
    assert rc == 0
    assert (out / "best.pt").is_file() and (out / "games.jsonl").is_file()
    run = json.loads((out / "run.json").read_text())
    assert run["status"] == "done" and set(run["eval"]) >= {"base", "final"}
    rows = [json.loads(l) for l in (out / "progress.jsonl").read_text().splitlines() if l.strip()]
    assert rows and rows[0]["examples"] > 0 and "eval" in rows[0]

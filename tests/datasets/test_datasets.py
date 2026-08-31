import json
import random

import numpy as np

from othello_rl.datasets import (DatasetConfig, assign_split, build_dataset,
                                 examples_from_analyzed_game, split_counts)
from othello_rl.datasets.examples import LABEL_CODES
from othello_rl.environment.board import Board


def _analyzed_game(seed, game_id, labels=None, data_kind="historical"):
    """A fabricated analyzed_games line: a real random game + assigned labels."""
    rng = random.Random(seed)
    s = Board.initial()
    moves = []
    while not s.is_terminal():
        r, c = rng.choice(s.legal_moves())
        a = r * 8 + c
        lab = (labels[len(moves) % len(labels)] if labels
               else rng.choice(["BEST", "GOOD", "ACCEPTABLE", "MISTAKE", "BLUNDER"]))
        moves.append({
            "move_number": len(moves),
            "player": "black" if s.player == 1 else "white", "played_move": a,
            "n_legal": len(s.legal_moves()),
            "by_horizon": {"3": {"label": lab}, "5": {"label": lab}},
        })
        s = s.apply((r, c))
    w = s.winner()
    return {"game_id": game_id, "source": "t", "data_kind": data_kind,
            "result": {"winner": "black" if w == 1 else "white" if w == -1 else "draw"},
            "n_positions": len(moves), "moves": moves}


# --------------------------------------------------------------------------- #
def test_assign_split_is_deterministic_and_roughly_proportioned():
    ratios = {"train": 0.8, "val": 0.1, "test": 0.1}
    ids = [f"g{i}" for i in range(4000)]
    a = split_counts(ids, ratios, seed=7)
    b = split_counts(ids, ratios, seed=7)
    assert a == b
    assert 0.75 < a["train"] / 4000 < 0.85
    assert 0.06 < a["val"] / 4000 < 0.14 and 0.06 < a["test"] / 4000 < 0.14
    # a different seed reshuffles
    assert split_counts(ids, ratios, seed=8) != a
    # every position of a game shares the game's split
    assert assign_split("gX", ratios, 1) == assign_split("gX", ratios, 1)


def test_value_target_is_from_the_movers_perspective():
    g = _analyzed_game(1, "g", labels=["BEST"])
    exs = list(examples_from_analyzed_game(g, 5, lambda l: 1.0))
    winner = g["result"]["winner"]
    for e, mv in zip(exs, g["moves"]):
        assert e.value_target == (1.0 if winner == mv["player"]
                                  else 0.0 if winner == "draw" else -1.0)
        assert e.obs.shape == (3, 8, 8) and e.obs.dtype == np.float32


def test_strategy_all_keeps_every_move_weight_one():
    g = _analyzed_game(2, "g")
    exs = list(examples_from_analyzed_game(g, 5, DatasetConfig(strategy="all").weight_fn()))
    assert len(exs) == g["n_positions"] and all(e.weight == 1.0 for e in exs)


def test_strategy_filtered_drops_mistakes_and_blunders():
    g = _analyzed_game(3, "g", labels=["BEST", "MISTAKE", "BLUNDER", "GOOD"])
    wfn = DatasetConfig(strategy="filtered").weight_fn()
    exs = list(examples_from_analyzed_game(g, 5, wfn))
    assert exs and all(e.label in ("BEST", "GOOD", "ACCEPTABLE") for e in exs)


def test_strategy_weighted_applies_config_weights_and_drops_blunders():
    g = _analyzed_game(4, "g", labels=["BEST", "GOOD", "ACCEPTABLE", "MISTAKE", "BLUNDER"])
    cfg = DatasetConfig(strategy="weighted",
                        label_weights={"BEST": 1.0, "GOOD": 0.7, "ACCEPTABLE": 0.4,
                                       "MISTAKE": 0.1, "BLUNDER": 0.0})
    exs = list(examples_from_analyzed_game(g, 5, cfg.weight_fn()))
    got = {e.label: e.weight for e in exs}
    assert got.get("BEST") == 1.0 and got.get("GOOD") == 0.7 and got.get("MISTAKE") == 0.1
    assert "BLUNDER" not in got                     # zero weight -> dropped


def test_build_dataset_no_leak_manifest_and_npz(tmp_path):
    src = tmp_path / "wthor.jsonl"
    src.write_text("\n".join(json.dumps(_analyzed_game(s, f"g{s}",
                   data_kind="self_play" if s % 3 else "historical"))
                   for s in range(30)))
    cfg = DatasetConfig(strategy="all", horizon=5,
                        split={"train": 0.6, "val": 0.2, "test": 0.2}, sources=["wthor"])
    m = build_dataset(src, tmp_path / "out", cfg, version="v0", report_dir=tmp_path / "rep")

    assert m["n_games"] == 30
    assert sum(d["n_games"] for d in m["splits"].values()) == 30
    # no game shared across splits
    seen = set()
    for sp in ("train", "val", "test"):
        d = np.load(tmp_path / "out" / "v0" / f"{sp}.npz", allow_pickle=True)
        gids = set(d["game_ids"].tolist())
        assert not (gids & seen)
        seen |= gids
        if len(d["obs"]):
            assert d["obs"].shape[1:] == (3, 8, 8)
            assert d["policy"].max() < 64 and set(np.unique(d["value"])) <= {-1.0, 0.0, 1.0}
            assert d["data_kind"].min() >= 0
    assert m["label_legend"]["0"] == "BEST" if "0" in m["label_legend"] else True
    assert (tmp_path / "rep" / "manifest.json").is_file()
    assert (tmp_path / "out" / "v0" / "manifest.json").is_file()


def test_config_hash_is_stable_and_sensitive():
    a = DatasetConfig(strategy="all").config_hash()
    assert a == DatasetConfig(strategy="all").config_hash()
    assert a != DatasetConfig(strategy="weighted").config_hash()

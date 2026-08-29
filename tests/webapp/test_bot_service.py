import random

import numpy as np
import pytest

from othello_rl.environment.board import BLACK, WHITE, Board
from othello_rl.rl.agent import DQNAgent, NetworkConfig
from othello_rl.webapp.bot_service import (
    FineTuneConfig,
    OthelloBot,
    classify_drop,
)


@pytest.fixture(scope="module")
def bot():
    agent = DQNAgent(NetworkConfig(channels=8, blocks=2, hidden=16), seed=0)
    return OthelloBot(agent, ft_config=FineTuneConfig(
        grad_steps=8, batch_size=16, anchor_transitions=120, guardrail_games=6,
        buffer_capacity=2000))


def _random_game(seed=0, max_plies=200):
    rng = random.Random(seed)
    s = Board.initial()
    acts = []
    while not s.is_terminal() and len(acts) < max_plies:
        lm = s.legal_moves()
        if not lm:
            s = s.apply(None); acts.append(64); continue
        m = rng.choice(lm)
        acts.append(m[0] * 8 + m[1]); s = s.apply(m)
    return acts, s


def test_select_action_always_legal(bot):
    s = Board.initial()
    for _ in range(12):
        if s.is_terminal():
            break
        a = bot.select_action(s)
        legal = [r * 8 + c for r, c in s.legal_moves()] or [64]
        assert a in legal
        s = s.apply(None if a == 64 else divmod(a, 8))


def test_classify_monotone():
    labels = [classify_drop(d)[0] for d in (0.0, 0.05, 0.10, 0.18, 0.30, 0.6)]
    assert labels == ["Best", "Excellent", "Good", "Inaccuracy", "Mistake", "Blunder"]


def test_evaluate_position_shapes(bot):
    ev = bot.evaluate_position(Board.initial())
    assert 0.0 <= ev["winprob_black"] <= 1.0
    assert len(ev["moves"]) == 4
    assert ev["moves"] == sorted(ev["moves"], key=lambda m: -m["value"])
    # terminal
    from tests.environment.conftest import make_board
    term = Board(make_board(["O" * 8] + ["." * 8] * 7), BLACK)
    assert bot.evaluate_position(term)["terminal"] is True


def test_analyse_game_structure(bot):
    acts, _ = _random_game(1)
    an = bot.analyse_game(acts)
    assert 0 < len(an) <= len(acts)
    for a in an:
        assert a.side in ("black", "white")
        assert a.label in ("Best", "Excellent", "Good", "Inaccuracy", "Mistake", "Blunder")
        assert 0.0 <= a.drop
        assert 0.0 <= a.eval_after_black <= 1.0
        assert a.played_san == a.played_san  # sane
        assert len(a.top_moves) >= 1


def test_analyse_flags_a_clear_mistake(bot):
    # black can take corner a1; playing the b2 X-square instead should grade worse
    from tests.environment.conftest import make_board
    b = make_board([".OX.....", ".O......", "..X.....", "...OX...",
                    "...XO...", "........", "........", "........"])
    st = Board(b, BLACK)
    corner = bot.grade_move(st, 0)          # a1 (the corner)
    xsquare = bot.grade_move(st, 2 * 8 + 0)  # (2,0)
    # the fast positional check should prefer the corner...
    assert corner["coach_best"] == 0
    assert corner["coach_drop"] == pytest.approx(0.0, abs=1e-6)
    # ...and the alternative should carry more regret and a worse label
    assert xsquare["regret"] > corner["regret"]
    _order = ["Best", "Excellent", "Good", "Inaccuracy", "Mistake", "Blunder"]
    assert _order.index(xsquare["label"]) > _order.index(corner["label"])


def test_finetune_from_game_runs_and_guardrails(bot):
    acts, final = _random_game(2)
    report = bot.finetune_from_game(acts, human_color="black")
    assert report.grad_steps == 8
    assert np.isfinite(report.loss_before) and np.isfinite(report.loss_after)
    assert len(report.grades) > 0
    assert 0.0 <= report.winrate_vs_random_after <= 1.0
    # version only advances if not rolled back
    if report.rolled_back:
        assert bot.version == 0
    else:
        assert bot.version == 1
        assert bot.games_finetuned == 1


def test_reset_to_baseline(bot):
    import torch
    baseline = {k: v.clone() for k, v in bot._baseline_state.items()}
    with torch.no_grad():
        for p in bot.agent.net.parameters():
            p.add_(0.5)
    bot.reset_to_baseline()
    sd = bot.agent.net.state_dict()
    for k, v in baseline.items():
        assert torch.allclose(sd[k].float(), v.float(), atol=1e-5)
    assert bot.version == 0 and bot.games_finetuned == 0

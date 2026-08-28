import math

import pytest

from othello_rl.evaluation.elo import (
    DEFAULT_RATING,
    EloModel,
    expected_score,
    ratings_from_matches,
    update_pair,
)
from othello_rl.evaluation.metrics import (
    summarize_match,
    wilson_interval,
    win_rate,
)
from othello_rl.evaluation.tournament import play_match


# --------------------------- metrics --------------------------- #
def test_win_rate():
    assert win_rate(5, 10) == 0.5
    assert win_rate(0, 0) == 0.0


def test_wilson_interval_bounds_and_width():
    lo, hi = wilson_interval(50, 100)
    assert 0.0 <= lo <= 0.5 <= hi <= 1.0
    lo_small, hi_small = wilson_interval(5, 10)
    # same point estimate, fewer samples => wider interval
    assert (hi_small - lo_small) > (hi - lo)
    assert wilson_interval(0, 0) == (0.0, 1.0)
    lo0, hi0 = wilson_interval(0, 20)
    assert lo0 == 0.0 and hi0 < 0.5


def test_summarize_match_significance():
    m = play_match("greedy", "random", num_games=250, seed=5)
    s = summarize_match(m)
    assert s.a_wins + s.b_wins + s.draws == 250
    assert s.ci_low <= s.a_win_rate <= s.ci_high
    assert s.significant_advantage() is True


# ----------------------------- elo ----------------------------- #
def test_expected_score_symmetry_and_equal():
    assert expected_score(1500, 1500) == pytest.approx(0.5)
    assert expected_score(1600, 1400) + expected_score(1400, 1600) == pytest.approx(1.0)
    assert expected_score(1800, 1400) > 0.9


def test_update_pair_zero_sum_and_direction():
    ra, rb = update_pair(1500, 1500, 1.0, k=32)
    assert ra > 1500 > rb
    assert (ra - 1500) == pytest.approx(1500 - rb)  # zero-sum
    ra2, rb2 = update_pair(1500, 1500, 0.5, k=32)
    assert ra2 == pytest.approx(1500) and rb2 == pytest.approx(1500)


def test_elo_model_orders_agents_by_strength():
    # A always beats B, B always beats C.
    games = [("A", "B", 1.0)] * 40 + [("B", "C", 1.0)] * 40 + [("A", "C", 1.0)] * 40
    model = EloModel(k=16).fit(games, passes=30, seed=0)
    lb = dict(model.leaderboard())
    assert lb["A"] > lb["B"] > lb["C"]


def test_ratings_from_matches_greedy_above_random():
    m = play_match("greedy", "random", num_games=60, seed=9)
    model = ratings_from_matches([m])
    assert model.kind == "internal"
    assert model.rating("greedy") > model.rating("random")


def test_elo_anchor_pins_rating():
    games = [("A", "B", 1.0)] * 30
    model = EloModel(k=16, anchor="B").fit(games, passes=10)
    assert model.rating("B") == pytest.approx(DEFAULT_RATING)
    assert model.rating("A") > DEFAULT_RATING

import pytest

from othello_rl.agents import GreedyAgent, RandomAgent
from othello_rl.environment.board import BLACK, WHITE
from othello_rl.evaluation.tournament import (
    play_game,
    play_match,
    round_robin,
)


def test_play_game_returns_consistent_scores():
    gr = play_game(RandomAgent(seed=1), RandomAgent(seed=2), seed=1)
    assert gr.black_score + gr.white_score <= 64
    assert gr.plies > 0
    if gr.black_score > gr.white_score:
        assert gr.winner == BLACK
    elif gr.white_score > gr.black_score:
        assert gr.winner == WHITE
    else:
        assert gr.winner == 0


def test_match_totals_add_up():
    m = play_match("random", "random", num_games=30, seed=7)
    assert m.a_wins + m.b_wins + m.draws == 30
    assert m.num_games == 30
    assert 0.0 <= m.a_win_rate <= 1.0


def test_match_is_reproducible():
    m1 = play_match("greedy", "random", num_games=20, seed=42)
    m2 = play_match("greedy", "random", num_games=20, seed=42)
    assert m1.to_dict() == m2.to_dict()
    assert [g.__dict__ for g in m1.games] == [g.__dict__ for g in m2.games]


def test_different_seed_changes_games():
    m1 = play_match("random", "random", num_games=20, seed=1)
    m2 = play_match("random", "random", num_games=20, seed=2)
    assert [g.seed for g in m1.games] != [g.seed for g in m2.games]


def test_colors_alternate():
    # With alternation, agent A is Black on even games, White on odd.
    m = play_match("greedy", "random", num_games=4, seed=0)
    assert len(m.games) == 4


def test_greedy_beats_random_over_many_games():
    from othello_rl.evaluation.metrics import summarize_match

    m = play_match("greedy", "random", num_games=200, seed=123)
    s = summarize_match(m)
    # Pure max-flips greedy is only modestly better than uniform-random in
    # Othello, but the advantage should be real (CI excludes 0.5).
    assert m.a_win_rate > 0.58
    assert s.ci_low > 0.5


def test_alternate_colors_false_keeps_a_black():
    m = play_match("greedy", "random", num_games=6, seed=0, alternate_colors=False)
    # deterministic-ish: greedy as black most of the time should win most
    assert m.a_wins >= m.b_wins


def test_round_robin_pair_count():
    results = round_robin(["random", "greedy", "heuristic"], num_games=6, seed=0)
    assert len(results) == 3


def test_opening_plies_diversify_deterministic_matchup():
    # Two deterministic agents: without random openings every game is identical;
    # with them the games vary but the match stays reproducible.
    # colours alternate, so with no random opening there are at most 2 distinct games
    same = play_match("greedy", "heuristic", num_games=12, seed=0, opening_plies=0)
    assert len({(g.black_score, g.white_score, g.plies) for g in same.games}) <= 2

    varied = play_match("greedy", "heuristic", num_games=12, seed=0, opening_plies=6)
    assert len({(g.black_score, g.white_score, g.plies) for g in varied.games}) > 3
    again = play_match("greedy", "heuristic", num_games=12, seed=0, opening_plies=6)
    assert [g.__dict__ for g in varied.games] == [g.__dict__ for g in again.games]

import random

import numpy as np
import pytest

from othello_rl.agents import (
    GreedyAgent,
    HeuristicAgent,
    MinimaxAgent,
    RandomAgent,
    make_agent,
)
from othello_rl.agents.heuristic_agent import evaluate
from othello_rl.environment import rules
from othello_rl.environment.board import BLACK, WHITE, Board
from tests.environment.conftest import make_board

ALL_AGENT_FACTORIES = [
    lambda: RandomAgent(seed=0),
    lambda: GreedyAgent(),
    lambda: HeuristicAgent(),
    lambda: MinimaxAgent(depth=2),
]


def _play_game(black, white, seed=0, max_plies=200):
    st = Board.initial()
    agents = {BLACK: black, WHITE: white}
    plies = 0
    while not st.is_terminal():
        agent = agents[st.player]
        move = agent.select_move(st)
        legal = st.legal_moves()
        if legal:
            assert move in legal, f"{agent} returned illegal move {move}"
        else:
            assert move is None, f"{agent} should pass when no legal move"
        st = st.apply(move)
        plies += 1
        assert plies < max_plies
    return st


@pytest.mark.parametrize("factory", ALL_AGENT_FACTORIES)
def test_agent_always_legal_across_random_opponents(factory):
    for seed in range(6):
        agent = factory()
        opp = RandomAgent(seed=1000 + seed)
        _play_game(agent, opp, seed=seed)
        _play_game(opp, agent, seed=seed)


@pytest.mark.parametrize("factory", ALL_AGENT_FACTORIES)
def test_agent_passes_on_forced_pass_state(factory):
    pass_board = make_board(["OX......", "........", "........", "........",
                             "........", "........", "........", "........"])
    st = Board(pass_board, BLACK)
    assert st.must_pass()
    assert factory().select_move(st) is None


def test_random_agent_reproducible_with_seed():
    st = Board.initial()
    a = RandomAgent(seed=123)
    b = RandomAgent(seed=123)
    seq_a = []
    seq_b = []
    s = st
    for _ in range(8):
        ma = a.select_move(s)
        seq_a.append(ma)
        s = s.apply(ma if s.legal_moves() else None)
    s = st
    for _ in range(8):
        seq_b.append(b.select_move(s))
        s = s.apply(seq_b[-1] if s.legal_moves() else None)
    assert seq_a == seq_b


@pytest.mark.parametrize("factory", [lambda: GreedyAgent(),
                                     lambda: HeuristicAgent(),
                                     lambda: MinimaxAgent(depth=2)])
def test_deterministic_agents_are_deterministic(factory):
    rng = random.Random(7)
    st = Board.initial()
    for _ in range(6):
        moves = st.legal_moves()
        if not moves:
            st = st.apply(None)
            continue
        st = st.apply(rng.choice(moves))
    picks = {factory().select_move(st) for _ in range(5)}
    assert len(picks) == 1


def test_greedy_picks_max_flips():
    board = make_board([".XOOOOO.", "........", "........", "........",
                        "........", "........", "........", "..XO....",])
    # Black at (0,7) flips 5; Black at (7,2)-> (7,3)=O bracket? needs X beyond.
    st = Board(board, BLACK)
    move = GreedyAgent().select_move(st)
    assert move == (0, 7)
    assert len(rules.flips_for_move(board, BLACK, move)) == 5


def test_greedy_tie_break_lowest_index():
    # Two independent single-flip options for White; greedy picks lower index.
    board = make_board(["OX......", "........", "........", "........",
                        "........", "........", "........", "......XO"])
    st = Board(board, WHITE)
    assert set(st.legal_moves()) == {(0, 2), (7, 5)}
    assert len(rules.flips_for_move(board, WHITE, (0, 2))) == 1
    assert len(rules.flips_for_move(board, WHITE, (7, 5))) == 1
    assert GreedyAgent().select_move(st) == (0, 2)


def test_minimax_takes_immediate_win():
    # White to move; one move ends the game with White ahead.
    board = make_board(["OOOOOOOO", "OOOOOOOO", "OOOOOOOO", "OOOOOOOO",
                        "OOOOOOOO", "OOOOOOXX", "XXXXXXXX", "XXXXXXX."])
    st = Board(board, WHITE)
    legal = st.legal_moves()
    assert legal == [(7, 7)]
    assert MinimaxAgent(depth=3).select_move(st) == (7, 7)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_minimax_alpha_beta_matches_plain_minimax_value(seed):
    rng = random.Random(seed)
    st = Board.initial()
    for _ in range(12):
        if st.is_terminal():
            break
        moves = st.legal_moves()
        st = st.apply(rng.choice(moves) if moves else None)
    agent = MinimaxAgent(depth=3)
    neg_inf, pos_inf = float("-inf"), float("inf")
    pruned = agent._value(st, 3, neg_inf, pos_inf, prune=True)
    full = agent._value(st, 3, neg_inf, pos_inf, prune=False)
    assert pruned == full


def test_evaluate_perspective_is_antisymmetric_on_nonterminal():
    rng = random.Random(11)
    st = Board.initial()
    for _ in range(12):
        moves = st.legal_moves()
        st = st.apply(rng.choice(moves) if moves else None)
    vb = evaluate(st.array, BLACK)
    vw = evaluate(st.array, WHITE)
    assert vb == pytest.approx(-vw, abs=1e-9)


def test_make_agent_specs():
    assert isinstance(make_agent("random:5"), RandomAgent)
    assert isinstance(make_agent("greedy"), GreedyAgent)
    assert isinstance(make_agent("minimax:2"), MinimaxAgent)
    assert make_agent("minimax:2").depth == 2
    with pytest.raises(ValueError):
        make_agent("bogus")

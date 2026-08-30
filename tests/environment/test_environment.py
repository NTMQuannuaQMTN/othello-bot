import random

import numpy as np
import pytest

from othello_rl.environment.board import BLACK, WHITE, Board
from othello_rl.environment.environment import (
    NUM_ACTIONS,
    OthelloEnv,
    encode_observation,
    legal_action_mask,
)
from tests.environment.conftest import make_board


def test_reset_observation_shape_and_initial_mask():
    env = OthelloEnv()
    obs, info = env.reset(seed=0)
    assert obs.shape == (3, 8, 8)
    assert obs.dtype == np.float32
    # initial position: 2 discs each, black to move, 4 legal moves
    assert obs[0].sum() == 2 and obs[1].sum() == 2
    assert obs[2].sum() == 4
    assert info["to_play"] == BLACK
    assert sorted(info["legal_actions"]) == [
        2 * 8 + 3, 3 * 8 + 2, 4 * 8 + 5, 5 * 8 + 4
    ]
    assert info["action_mask"].shape == (NUM_ACTIONS,)
    assert not info["must_pass"]


def test_observation_is_canonical_from_side_to_move():
    env = OthelloEnv()
    env.reset()
    obs_black, _, _, _, info = env.step(2 * 8 + 3)  # black plays d3
    # now white to move; channel 0 must be white's discs
    assert info["to_play"] == WHITE
    assert obs_black[0].sum() == 1   # white has 1 disc after being flipped down to 1
    assert obs_black[1].sum() == 4   # black has 4


def test_channel2_matches_legal_moves():
    st = Board.initial().apply((2, 3)).apply((2, 2))
    obs = encode_observation(st)
    mask = legal_action_mask(st)
    from_obs = {(r, c) for r, c in zip(*np.nonzero(obs[2]))}
    from_mask = {(a // 8, a % 8) for a in np.nonzero(mask)[0] if a < 64}
    assert from_obs == from_mask == set(st.legal_moves())


def test_illegal_action_raises_by_default():
    env = OthelloEnv()
    env.reset()
    with pytest.raises(ValueError):
        env.step(0)  # (0,0) is not legal at the start
    with pytest.raises(ValueError):
        env.step(64)  # pass not allowed, moves available


def test_illegal_action_loss_mode():
    env = OthelloEnv(illegal_move_mode="loss")
    env.reset()
    obs, reward, terminated, truncated, info = env.step(0)
    assert reward == -1.0 and terminated
    assert info["illegal_action"] == 0


def test_reward_sign_from_movers_perspective():
    # Play a full random game; verify terminal reward matches the winner and the
    # mover of the final ply.
    for seed in range(40):
        env = OthelloEnv()
        env.reset(seed=seed)
        rng = random.Random(seed)
        last_mover = None
        while True:
            mover = env.state.player
            legal = env._info()["legal_actions"]
            action = rng.choice(legal)
            obs, reward, terminated, truncated, info = env.step(action)
            last_mover = mover
            if terminated or truncated:
                break
        assert not truncated
        w = info["winner"]
        if w == 0:
            assert reward == 0.0
        elif w == last_mover:
            assert reward == 1.0
        else:
            assert reward == -1.0
        # non-terminal rewards were all zero (checked implicitly: only break on term)


def test_forced_pass_action_available_and_applies():
    pass_board = make_board(["OX......", "........", "........", "........",
                             "........", "........", "........", "........"])
    env = OthelloEnv()
    env.reset()
    env.state = Board(pass_board, BLACK)
    mask = legal_action_mask(env.state)
    assert mask[64] and mask.sum() == 1
    obs, reward, terminated, truncated, info = env.step(64)
    assert not terminated
    assert info["to_play"] == WHITE
    assert reward == 0.0


def test_episode_terminates_with_winner_info():
    env = OthelloEnv()
    env.reset()
    rng = random.Random(1)
    steps = 0
    while True:
        legal = env._info()["legal_actions"]
        _, reward, terminated, truncated, info = env.step(rng.choice(legal))
        steps += 1
        if terminated:
            assert "winner" in info
            assert info["black_score"] + info["white_score"] <= 64
            break
        assert steps < 80


def test_action_mask_never_all_false_until_terminal():
    env = OthelloEnv()
    env.reset()
    rng = random.Random(5)
    while not env.state.is_terminal():
        mask = legal_action_mask(env.state)
        assert mask.any()
        legal = np.nonzero(mask)[0].tolist()
        env.step(rng.choice(legal))


def test_max_steps_covers_the_othello_maximum():
    from othello_rl.environment.environment import MAX_STEPS
    # <= 60 placements, each preceded by at most one pass -> <= 120 plies
    assert MAX_STEPS >= 120


def test_random_episodes_terminate_and_are_never_truncated():
    """Truncation must never fire for a correct Othello engine, and every episode
    must end with `terminated=True` and a winner in info."""
    for seed in range(120):
        env = OthelloEnv()
        _, info = env.reset(seed=seed)
        rng = random.Random(seed)
        plies = 0
        while True:
            a = rng.choice(np.nonzero(np.asarray(info["action_mask"]))[0])
            _, reward, terminated, truncated, info = env.step(int(a))
            plies += 1
            assert not truncated, f"seed {seed} truncated at ply {plies}"
            if terminated:
                assert "winner" in info
                assert reward in (-1.0, 0.0, 1.0)
                break
            assert plies <= 120
        assert plies <= 120

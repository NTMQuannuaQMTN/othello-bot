"""Potential-based corner-safety shaping (``rl/shaping.py``)."""
import numpy as np

from othello_rl.environment.board import BLACK, WHITE, Board
from othello_rl.rl.opponents import FixedOpponentEnv
from othello_rl.rl.shaping import CornerShaping


def _empty():
    return np.zeros((8, 8), dtype=np.int8)


def test_potential_rewards_corners_penalises_x_squares():
    sh = CornerShaping()
    b = _empty()
    assert sh.potential(b, BLACK) == 0.0

    b[1, 1] = BLACK                      # black on b2, a1 still empty -> bad for black
    assert sh.potential(b, BLACK) < 0
    assert sh.potential(b, WHITE) > 0    # ... and good for white

    b[0, 0] = BLACK                      # black now holds a1 -> the b2 liability clears
    assert sh.potential(b, BLACK) == CornerShaping().corner_weight


def test_delta_is_potential_based():
    sh = CornerShaping(gamma=0.9)
    b0, b1 = _empty(), _empty()
    b1[1, 1] = BLACK                     # black plays the X-square
    # F = γ·Φ(s') − Φ(s)
    assert sh.delta(b0, b1, BLACK, done=False) == \
        0.9 * sh.potential(b1, BLACK) - sh.potential(b0, BLACK)
    # Φ(terminal) is taken as 0
    assert sh.delta(b0, b1, BLACK, done=True) == -sh.potential(b0, BLACK)
    # a no-op transition just discounts the potential
    assert sh.delta(b1, b1, BLACK, done=False) == \
        (0.9 - 1.0) * sh.potential(b1, BLACK)


def test_curriculum_runs_with_shaping(tmp_path):
    from othello_rl.rl.agent import DQNAgent, NetworkConfig
    from othello_rl.rl.curriculum import CurriculumConfig, Stage, run_curriculum
    from othello_rl.rl.trainer import DQNConfig

    agent = DQNAgent(NetworkConfig(channels=8, blocks=2, hidden=16), seed=0)
    cfg = CurriculumConfig(
        stages=[Stage("s1", "random", env_steps=250)],
        eval_opponents={"random": "random"}, eval_games=4, eval_every=200,
        checkpoint_every=200,
        dqn=DQNConfig(batch_size=16, buffer_capacity=600, warmup_steps=30,
                      target_sync=20, epsilon_decay_steps=200),
        shaping=CornerShaping(),
    )
    run_curriculum(agent, cfg, tmp_path, seed=0)
    assert (tmp_path / "checkpoints" / "final.pt").is_file()


def test_from_config():
    assert CornerShaping.from_config(None).enabled is False
    assert CornerShaping.from_config({}).enabled is False
    s = CornerShaping.from_config({"x_square_weight": 0.3}, gamma=0.95)
    assert s.enabled and s.x_square_weight == 0.3 and s.gamma == 0.95


def test_fixed_opponent_env_applies_shaping():
    base = FixedOpponentEnv("random", learner_color=BLACK, seed=0)
    shaped = FixedOpponentEnv("random", learner_color=BLACK, seed=0,
                              shaping=CornerShaping())
    assert base.shaping is None and shaped.shaping is not None

    obs, info = shaped.reset(seed=1)
    total = 0.0
    for _ in range(200):
        legal = np.nonzero(info["action_mask"])[0]
        obs, r, term, trunc, info = shaped.step(int(legal[0]))
        total += r
        assert np.isfinite(r)
        if term or trunc:
            break
    assert term or trunc
    assert abs(total) < 6.0            # terminal ±1 plus bounded shaping


def test_disabled_shaping_is_a_no_op():
    # enabled: false -> env.shaping stays None -> identical rewards to no shaping
    off = FixedOpponentEnv("random", learner_color=BLACK, seed=3,
                           shaping=CornerShaping(enabled=False))
    plain = FixedOpponentEnv("random", learner_color=BLACK, seed=3)
    assert off.shaping is None and plain.shaping is None
    _, i_off = off.reset(seed=3)
    _, i_plain = plain.reset(seed=3)
    a = int(np.nonzero(i_off["action_mask"])[0][0])
    assert off.step(a)[1] == plain.step(a)[1]

import numpy as np
import pytest

from othello_rl.environment.board import BLACK, WHITE
from othello_rl.rl.agent import DQNAgent, NetworkConfig
from othello_rl.rl.opponents import FixedOpponentEnv
from othello_rl.rl.trainer import DQNConfig, DQNTrainer


def _agent():
    return DQNAgent(NetworkConfig(channels=16, blocks=2, hidden=32), seed=0)


def test_opening_plies_diversify_start_states():
    seen = set()
    env = FixedOpponentEnv("random", learner_color=BLACK, seed=0, opening_plies=6)
    for _ in range(15):
        obs, info = env.reset()
        seen.add(obs[:2].tobytes())
        # after opening + opponent replies it is the learner's turn (or terminal)
        assert env.env.state.player == BLACK or env.env.state.is_terminal()
    assert len(seen) > 5  # genuinely varied openings


def test_fixed_opponent_env_gives_learner_turn_and_perspective():
    env = FixedOpponentEnv("random", learner_color=WHITE, seed=0)
    obs, info = env.reset(seed=0)
    # learner is white; after black's opening move it's white's turn
    assert env.env.state.player == WHITE
    assert info["learner_color"] == WHITE
    assert obs.shape == (3, 8, 8)
    assert info["action_mask"][:64].sum() >= 1


def test_fixed_opponent_env_episode_terminates_with_pm1_reward():
    for seed in range(20):
        env = FixedOpponentEnv("greedy", learner_color="random", seed=seed)
        obs, info = env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        total_reward = 0.0
        steps = 0
        while True:
            legal = np.nonzero(info["action_mask"])[0]
            assert len(legal) >= 1
            a = int(rng.choice(legal))
            obs, reward, terminated, truncated, info = env.step(a)
            total_reward += reward
            steps += 1
            if terminated or truncated:
                break
            assert steps < 80
        assert reward in (-1.0, 0.0, 1.0)
        assert total_reward in (-1.0, 0.0, 1.0)  # sparse: only terminal is non-zero


def test_trainer_smoke_runs_and_loss_is_finite():
    env = FixedOpponentEnv("random", learner_color="random", seed=0)
    agent = _agent()
    cfg = DQNConfig(batch_size=32, buffer_capacity=2000, warmup_steps=100,
                    target_sync=50, epsilon_decay_steps=500)
    trainer = DQNTrainer(env, agent, cfg, seed=0)
    trainer.learn(total_env_steps=800, log_every=200)

    assert trainer.env_steps >= 800
    assert trainer.train_steps > 0
    assert trainer.episodes > 0
    assert len(trainer.buffer) > 0
    losses = [r["loss"] for r in trainer.metrics.history if "loss" in r]
    assert losses and all(np.isfinite(l) for l in losses)
    # agent meta is synced for checkpointing
    assert agent.meta.env_steps == trainer.env_steps


def test_training_is_deterministic_given_seed():
    def run():
        from othello_rl.utils.seed import seed_everything
        seed_everything(123)
        ag = DQNAgent(NetworkConfig(channels=8, blocks=2, hidden=16), seed=123)
        env = FixedOpponentEnv("random", learner_color="random", seed=123, opening_plies=4)
        tr = DQNTrainer(env, ag, DQNConfig(batch_size=16, buffer_capacity=1000,
                                           warmup_steps=50, target_sync=25,
                                           epsilon_decay_steps=400), seed=123)
        tr.learn(700, log_every=700)
        import torch
        with torch.no_grad():
            return float(sum(p.abs().sum() for p in ag.net.parameters()))
    assert run() == run()


def test_epsilon_schedule_monotone_decreasing():
    cfg = DQNConfig(epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_steps=1000)
    vals = [cfg.epsilon(s) for s in range(0, 1500, 100)]
    assert vals[0] == 1.0
    assert all(b <= a + 1e-9 for a, b in zip(vals, vals[1:]))
    assert vals[-1] == pytest.approx(0.05)


def test_trainer_checkpoint_after_training(tmp_path):
    env = FixedOpponentEnv("random", learner_color=BLACK, seed=1)
    agent = _agent()
    cfg = DQNConfig(batch_size=16, buffer_capacity=1000, warmup_steps=50,
                    target_sync=25, epsilon_decay_steps=200)
    trainer = DQNTrainer(env, agent, cfg, seed=1)
    trainer.learn(total_env_steps=300, log_every=100)
    p = tmp_path / "trained.pt"
    agent.save(p)
    reloaded = DQNAgent.from_checkpoint(p)
    assert reloaded.meta.env_steps == trainer.env_steps

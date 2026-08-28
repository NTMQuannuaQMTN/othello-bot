# RL algorithm choice — Phase 5

## Decision

The first learning agent is a **masked Deep Q-Network (DQN)** with:

- a target network (hard periodic copy),
- a uniform experience replay buffer,
- epsilon-greedy exploration over **legal** actions only,
- illegal actions masked to `-inf` before every `max` / `argmax`,
- Double-DQN target (action selected by the online net, evaluated by the target
  net) — one line of code, meaningfully reduces value over-estimation.

## Why DQN first (not policy gradient, not AlphaZero)

| Requirement | How DQN fits |
|---|---|
| Discrete action space (65) | Native — one Q-value per action. |
| Illegal-action masking | Trivial and exact: set masked Q to `-inf`. No renormalisation needed, unlike a softmax policy. |
| Turn-based, zero-sum, sparse reward | Handled by playing the opponent inside an env wrapper and bootstrapping; reward `±1` at terminal only. We store transitions from the learner's perspective and negate value across the opponent ply (`r + γ·(−1)·maxQ(s')`), or simpler: wrap a fixed opponent so the env is a stationary single-agent MDP for the learner (Phase 6). |
| Sample efficiency on a tiny board | Replay + bootstrapping reuses data well; an 8×8 board with a small conv net trains on CPU. |
| No differentiable opponent / no model needed | DQN is model-free and off-policy. |
| Must be *verifiable* quickly | Clear success metric: greedy-eval win rate vs Random must rise well above the untrained net, reproducibly. |

Policy-gradient / actor-critic (REINFORCE, PPO) are viable but on-policy and
higher-variance with purely sparse rewards; masking a categorical policy is
slightly fiddlier (renormalise over legal actions). We keep PG as a comparison
point later.

AlphaZero (policy+value+MCTS, self-play targets) is the *long-term* target
(§13 of the spec) and is deliberately deferred until the simple pipeline is
validated end-to-end.

## Opponent handling for the MDP

Phase 6 trains against **fixed opponents**. The `FixedOpponentEnv` wrapper
(`rl/opponents.py`) wraps `OthelloEnv`:

- the learner is assigned a colour (fixed or randomised per episode),
- whenever it is the opponent's turn (including forced passes) the wrapper plays
  the opponent's move internally,
- `step()` therefore returns the next state where it is *the learner's* turn (or
  terminal), and reward is from the learner's perspective.

This makes the environment a stationary MDP, so vanilla DQN convergence
arguments apply. Self-play (Phase 7) reintroduces a moving opponent via an
opponent pool.

## Network

`SmallOthelloNet`: 3×(3×3 conv, 32–64 ch, BN, ReLU) → flatten → MLP → 65 logits
(Q-values). ~50–100k params. Rationale: 8×8 is small; convolution captures local
flip structure and corner/edge patterns; depth kept low for CPU speed. A shared
torso with a value head is left as a hook for the AlphaZero phase.

## Hyperparameters (starting point, in `configs/train.yaml`)

γ=0.99 (episodes are short; could be 1.0), lr=1e-3 Adam, batch 256, replay 100k,
target sync every 1000 gradient steps, ε 1.0→0.05 over first ~30k steps, train
1 gradient step per env step after a 5k warmup.

These are starting values; Phase 6 tunes them against the "beats Random
reproducibly" criterion.

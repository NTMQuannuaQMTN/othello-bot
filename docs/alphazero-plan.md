# Future phase — AlphaZero-inspired upgrade (NOT yet implemented)

This is a design note only. Do not start until Phases 1–9 are complete and the
simple DQN pipeline is evaluated and documented.

## What already supports it

- `SmallOthelloNet` has a **value head** (`with_value_head=True`, `tanh` output)
  alongside the action head. For AlphaZero the action head is reinterpreted as a
  **policy** (softmax over the 65 masked logits) rather than Q-values.
- `OthelloEnv` gives a canonical `(3,8,8)` observation, a 65-way action space with
  an exact legal mask, and a terminal `±1` outcome — exactly the signal MCTS
  self-play needs.
- `Board` is immutable and cheap to copy (`Board.apply` returns a new state),
  which is what an MCTS tree needs for node expansion.
- `evaluation/` (tournaments, Elo, `evaluate_agent`) and `rl/self_play.OpponentPool`
  can be reused unchanged for the "gate a new net against the previous best" step.

## Components to add

| Component | Sketch |
|---|---|
| `rl/az_network.py` | Shared conv torso → (policy logits 65, value scalar). Reuse `ConvBlock`; likely add residual blocks + a small SE or global-pool for board-wide context. |
| `rl/mcts.py` | PUCT search over `Board` nodes: `N, W, Q, P` per edge; `select → expand (one net eval) → backup`. Dirichlet noise at the root; temperature-controlled move sampling. Masked priors (renormalise over legal actions). |
| `rl/az_selfplay.py` | Generate games with MCTS; store `(obs, π = visit-count distribution, z = game outcome)` triples. |
| `rl/az_trainer.py` | Loss = `(z − v)² − πᵀ log p + c‖θ‖²`. Sample from a replay of recent self-play games. |
| `rl/az_arena.py` | Play candidate vs current-best over N games with low temperature; promote if win rate > ~55%. |
| `scripts/az_train.py` | Iteration loop: self-play → train → arena-gate → repeat; checkpoints + `tracking.py` curves. |

## Milestones

1. MCTS + a *fixed* (heuristic-initialised or random) evaluator beats Greedy — validates the search.
2. One full self-play→train→gate iteration runs end to end and produces data.
3. Iterated training shows rising internal Elo vs the frozen baselines and vs
   previous best checkpoints (reuse `scripts/track.py`).

## Explicitly out of scope for now

Distributed self-play, GPU batching of MCTS leaf evals, learned-model (MuZero)
variants. CPU-only single-process is the target until it demonstrably learns.

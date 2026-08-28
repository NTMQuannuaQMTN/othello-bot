# Task checklist

Legend: `[ ]` todo · `[x]` done · `[~]` in progress

Tasks are marked `[x]` only after the implementation exists, its tests exist and
pass, and the full suite is green.

## Phase 0 — Project setup

- [x] Inspect repo (was empty)
- [x] Decide stack / Python version constraint (only 3.9.6 available; documented in PROJECT_SPEC)
- [x] Create `pyproject.toml`, `requirements.txt`, `.gitignore`
- [x] Create `README.md`, `PROJECT_SPEC.md`, `TASKS.md`, `PROGRESS.md`
- [x] Create `src/` layout package skeleton
- [x] Verify `pytest` runs

## Phase 1 — Othello game engine

- [x] Define board representation (constants, directions) — `environment/board.py`
- [x] Implement initial board
- [x] Human/flat coordinate conversion helpers
- [x] Implement legal move generation (`rules.legal_moves`)
- [x] Implement flip computation for a candidate move (all 8 directions)
- [x] Implement move execution (place + flip + turn switch)
- [x] Implement illegal-move rejection
- [x] Implement forced-pass logic
- [x] Implement game termination + score + winner
- [x] `Board` class wrapping the above
- [x] Tests: initial state
- [x] Tests: horizontal / vertical / diagonal captures
- [x] Tests: multi-direction capture, corner, edge, multi-piece
- [x] Tests: illegal moves (occupied, no-capture, out-of-range)
- [x] Tests: pass behaviour
- [x] Tests: termination, full board, winner, score
- [x] Property/fuzz test: random playouts never crash, invariants hold
- [x] Run full suite green

## Phase 2 — Baseline agents

- [ ] `agents/base.py` — `Agent` interface
- [ ] `RandomAgent`
- [ ] `GreedyAgent` (max immediate flips, deterministic tie-break)
- [ ] `HeuristicAgent` (configurable weights)
- [ ] `MinimaxAgent` (alpha-beta, configurable depth)
- [ ] Tests: all agents always return legal moves
- [ ] Tests: agents handle pass-only states
- [ ] Tests: deterministic agents reproducible
- [ ] Tests: minimax picks obvious win / corner; alpha-beta value == plain minimax

## Phase 3 — Baseline evaluation system

- [ ] `evaluation/tournament.py` — play_match / round-robin, alternating colours, seeds
- [ ] `evaluation/metrics.py` — W/L/D, mean score diff, Wilson CI
- [ ] `evaluation/elo.py` — internal Elo (labelled "internal")
- [ ] Machine-readable results (JSON)
- [ ] `scripts/evaluate.py` + `configs/evaluation.yaml`
- [ ] Performance report generator (markdown + plot)
- [ ] Tests for tournament/metrics/elo

## Phase 4 — RL environment

- [ ] `environment/environment.py` — `OthelloEnv` gym-like API
- [ ] Canonical `(3,8,8)` observation + length-65 action mask
- [ ] Sparse reward, done, info
- [ ] Seeding
- [ ] Tests: observation, mask, reward sign, termination, illegal action

## Phase 5 — Initial deep-RL agent

- [ ] `docs/rl-algorithm.md` — justify DQN
- [ ] `rl/network.py` — small conv net, Q-head length 65
- [ ] `rl/replay_buffer.py`
- [ ] `rl/agent.py` — masked epsilon-greedy / greedy eval, save/load
- [ ] `rl/trainer.py` — DQN loop, target net, metrics
- [ ] Tests: network shapes, masking, buffer, checkpoint round-trip, trainer smoke

## Phase 6 — Training vs fixed opponents

- [ ] `rl/opponents.py` fixed-opponent env wrapper
- [ ] `scripts/train.py` + `configs/train.yaml`
- [ ] Stage 1: vs Random — reproducible win-rate gain over untrained net
- [ ] Periodic eval + checkpointing + metrics JSON + plot
- [ ] Stage 2: vs Random+Greedy
- [ ] Stage 3: vs Heuristic
- [ ] Document results

## Phase 7 — Self-play

- [ ] `rl/self_play.py` — opponent pool, configurable sampling
- [ ] Checkpoint pool management
- [ ] Anti-forgetting eval vs historical checkpoints
- [ ] Tests: pool sampling distribution, snapshot/restore

## Phase 8 — Evaluation & experiment tracking

- [ ] Standard eval protocol vs Random/Greedy/Heuristic/Minimax/historical
- [ ] `utils/logging.py` metric logger (JSONL + CSV)
- [ ] Plots: games vs win-rate, iterations vs baseline perf, checkpoint vs Elo
- [ ] Experiment metadata capture

## Phase 9 — Human play interface

- [ ] `scripts/play.py` terminal UI

## Future — AlphaZero-inspired

- [ ] Policy+Value net, MCTS, visit-count targets, iterative self-play, eval gating

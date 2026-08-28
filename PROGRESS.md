# Progress

_Last updated: 2026-08-28_

## Current phase
Phase 6 — Training vs fixed opponents (curriculum).

## Current task
Write `scripts/train.py` + `configs/train.yaml`; run Stage 1 (vs Random) and
verify a reproducible win-rate gain over an untrained network.

## Completed tasks
- Phase 0: scaffolding, docs, deps.
- Phase 1: Othello engine + 31 tests (captures, illegal, pass, termination, fuzz).
  Move generation later switched to a fast short-circuiting scalar path
  (`rules._has_bracket` over `board.tolist()`); a numpy `legal_move_mask` exists
  and is cross-checked against the scalar reference.
- Phase 2: RandomAgent, GreedyAgent, HeuristicAgent (configurable weighted eval),
  MinimaxAgent (negamax + alpha-beta + static move ordering, pass-aware). 18 tests.
- Phase 3: `evaluation/` — tournament (`play_game`/`play_match`/`round_robin` with
  reproducible per-game seeds and random opening plies to diversify deterministic
  matchups), metrics (Wilson CI), internal Elo (`EloModel`, clearly labelled
  "internal"), JSON+markdown+PNG report, `scripts/evaluate.py`,
  `configs/evaluation.yaml`. Tests for all.
- Phase 4: `environment/environment.py` `OthelloEnv` — canonical (3,8,8) obs,
  length-65 action mask (64 = pass), sparse ±1 reward from mover's perspective,
  `illegal_move_mode` raise/loss. 9 tests.
- Phase 5: `docs/rl-algorithm.md` (justifies masked Double-DQN). `rl/network.py`
  (`SmallOthelloNet`, conv torso + Q head + optional value head, `masked_q`),
  `rl/replay_buffer.py` (uniform replay storing next-state masks), `rl/agent.py`
  (`DQNAgent`: masked eps-greedy + greedy eval, implements baseline `Agent`,
  checkpoint save/load/from_checkpoint, `clone_network`), `rl/opponents.py`
  (`FixedOpponentEnv` -> stationary single-agent MDP), `rl/trainer.py`
  (`DQNTrainer`: Double-DQN, target sync, Huber loss, eps schedule, metrics,
  optional periodic eval hook). 16 RL tests incl. trainer smoke + checkpoint.

## Test status
`python3 -m pytest` -> 94 passed (~21s).

## Latest baseline evaluation (experiments/, seed 20260828, 150 games/matchup,
   opening_plies=4)
A previous run (no opening randomisation) gave:
  greedy>random 0.58 · heuristic>random 0.94 · heuristic>greedy 1.00 ·
  minimax2~heuristic 0.50 · minimax3/4>heuristic 1.00 · minimax3>greedy 1.00.
  Internal Elo: minimax3 2477, minimax4 2469, minimax2 1323, heuristic 1301,
  random 793, greedy 637.
A refreshed run with opening_plies=4 is in progress; see experiments/<ts>_eval/.

## Latest training result
None yet — Phase 6 next.

## Known issues
- Python 3.9.6 only; deps `--user`. Use `python3 -m pytest`.
- Pure-Python minimax: depth 4 ~1.5 s/game. Fine for one-off evals, kept out of
  the fast test path.
- `greedy` internal Elo below `random`: pure max-flips greedy really is weak in
  Othello and the baseline matchup graph is sparse; not a bug.

## Next action
Phase 6: `scripts/train.py`, `configs/train.yaml`, Stage 1 vs Random with
periodic eval + checkpoints + win-rate plot; then Stages 2 (Random+Greedy) and 3
(Heuristic). Document results here and under `experiments/`.

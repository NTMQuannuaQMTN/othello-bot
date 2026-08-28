# Progress

_Last updated: 2026-08-28_

## Current phase
Phase 6 — running the tuned DQN curriculum and documenting results.
(Phases 7–9 code + tests complete; empirical self-play run still pending.)

## Current task
A full 3-stage curriculum run is executing in the background
(`experiments/20260828-201918_dqn_curriculum/`, ~380k env steps). When it
finishes: run `scripts/track.py` on its checkpoints, record win-rate/Elo curves,
verify the "beats Random reproducibly" criterion, then do a self-play run.

## Completed tasks
- Phase 0–5: see git history. Engine + baselines + evaluation framework + RL env
  + masked Double-DQN, all tested.
- Phase 3 baseline eval (real run, `experiments/20260828-194501_eval/`): sensible
  Elo ordering minimax4 2372 > minimax3 1804 > minimax2 1597 > heuristic 1445 >
  greedy 937 > random 846; heuristic beats random 0.96, minimax2 beats heuristic
  0.78, minimax4 beats heuristic 1.00.
- Phase 6 code: `rl/curriculum.py`, `scripts/train.py`, `configs/train.yaml`;
  untrained-baseline anchor eval, periodic eval + checkpoint + JSONL metrics,
  win-rate/return plots, `summary.md`.
  - Stage-1 sanity run (60k steps, small net): win rate vs Random 0.51 → 0.66,
    vs Greedy 0.43 → 0.64 (noisy). Added opening randomisation to training to
    align with the opening-randomised eval; tuned hyperparameters; larger run
    in progress.
- Phase 7 code: `rl/self_play.py` — `OpponentPool` (baseline / historical /
  recent, configurable distribution), `run_self_play` with snapshotting +
  anti-forgetting eval vs historical snapshots. Tested (smoke + unit).
- Phase 8 code: `evaluation/tracking.py` + `scripts/track.py` — per-checkpoint
  win rate vs baselines, round-robin internal Elo, win-rate/Elo-vs-step plots,
  `tracking.md/json`. `utils/logging.py` (JSONL+CSV), `utils/experiment.py`
  (git commit + versions metadata). Tested.
- Phase 9: `scripts/play.py` terminal UI (choose colour, baseline or checkpoint
  opponent, shows board + legal moves). Smoke-tested.

## Test status
`python3 -m pytest` → 110 passed (~35–40 s; slower while training runs).

## Latest training result
Stage-1 sanity (see above). Full curriculum run in progress — results pending.

## Known issues
- Python 3.9.6 only; deps `--user`; run `python3 -m pytest`.
- CPU-only, pure-Python engine + conv net → ~100–120 env steps/s. Training runs
  are minutes–hours; configs are sized accordingly.
- DQN eval win rates are noisy at 80–100 games/eval; trends over several evals
  are what matter, not single points.

## Next action
1. Wait for the curriculum run; run `scripts/track.py --run <dir>`.
2. Write `experiments/<dir>/RESULTS.md` with the win-rate/Elo curves and a
   reproducibility check (rerun stage 1 with a 2nd seed).
3. Do a `run_self_play` run from the curriculum's final checkpoint; check the
   anti-forgetting metrics.
4. Mark Phase 6/7 empirical tasks done in TASKS.md once documented.

# Progress

_Last updated: 2026-08-28_

## Current phase
Phase 2 — Baseline agents

## Current task
Implement `agents/base.py` interface + Random/Greedy/Heuristic/Minimax agents and tests.

## Completed tasks
- Phase 0: project scaffolding, docs, dependency setup, `pytest` runs.
- Phase 1: Othello engine complete.
  - `environment/board.py`: representation, constants, 8 directions, coord helpers,
    `Board` state class (immutable array + side to move, `apply`/`legal_moves`/
    `legal_actions`/`must_pass`/`is_terminal`/`winner`/`scores`/`render`).
  - `environment/rules.py`: `flips_for_move`, `legal_moves`, `has_any_move`,
    `apply_move`, `next_player` (handles forced-pass + consecutive move),
    `is_terminal`, `score`, `winner`.
  - 31 tests: initial state, H/V/diagonal/multi-direction/corner/edge/multi-piece
    captures, illegal (occupied/no-capture/out-of-range/illegal-pass), forced pass,
    termination (full board / stuck / draw), 300 random-playout fuzz with
    invariant checks. All green.

## Test status
`python3 -m pytest` → 31 passed (~6s; the 300-game fuzz dominates runtime).

## Latest training result
None (no RL yet).

## Known issues
- Reference machine only has Python 3.9.6. Deps installed `--user` into
  `~/Library/Python/3.9`; put its `bin/` on PATH or use `python3 -m pytest`.
- Pure-Python rules are ~correct-but-slow; acceptable per "correctness first".
  Revisit with bitboards only if RL throughput demands it.

## Next action
Phase 2: agent interface + 4 baseline agents + tests.

# Progress

_Last updated: 2026-08-29_

## Current phase
Phases 1–9 complete. Phase 7 self-play empirical run in progress; then optional
tuning / AlphaZero upgrade (future phase).

## Current task
Self-play run (`scripts/selfplay.py`, warm-started from the curriculum
`final.pt`) is executing. When it finishes: fill in section 3 of
`experiments/RESULTS.md` with its anti-forgetting metrics.

## Completed tasks
- **Phase 0–5** — scaffolding, tested Othello engine, 4 baseline agents,
  evaluation framework (tournaments / Wilson CI / internal Elo / reports),
  RL env, masked Double-DQN (network / replay / agent / trainer). See git history.
- **Phase 3 baseline eval** (`experiments/20260828-194501_eval/`): Elo ordering
  minimax4 2372 > minimax3 1804 > minimax2 1597 > heuristic 1445 > greedy 937 >
  random 846; full table in `experiments/RESULTS.md`.
- **Phase 6 — DQN curriculum, VALIDATED**
  (`experiments/20260828-201918_dqn_curriculum/`, 380k env steps):
  - Untrained → trained: vs Random 0.55 → 0.82, vs Greedy 0.41 → 0.82,
    vs Heuristic 0.05 → 0.22. Internal-Elo curve 1585 → ~1910.
  - Success criterion (beat Random reproducibly) met on 2 seeds
    (`20260829-002434_repro_s777`: 0.27 → 0.71). Training is deterministic given a
    seed (unit-tested).
  - Artifacts: `winrate_vs_steps.png`, `train_return.png`, `tracking/` plots,
    18 checkpoints, `metrics.jsonl`, `summary.md`.
- **Phase 7 code** — `OpponentPool` + `run_self_play` (anti-forgetting eval),
  `scripts/selfplay.py`, `configs/selfplay.yaml`. Tested.
- **Phase 8** — `scripts/track.py` (per-checkpoint win rate + round-robin
  internal Elo + curves); `utils/logging.py`, `utils/experiment.py`,
  `utils/progress.py`.
- **Phase 9** — `scripts/play.py` terminal UI.
- **Progress bars** — `train.py` / `selfplay.py` show a continuous tqdm bar
  (`--progress auto|on|off`); plain `[stage] N/total (pct%)` lines when redirected.

## Test status
`python3 -m pytest` → 117 passed (~30–60 s depending on background load).

## Known issues
- Python 3.9.6 only; deps `--user`; run `python3 -m pytest`.
- CPU-only, pure-Python engine → ~30–120 env steps/s depending on eval cadence;
  a full curriculum run is ~3–4 h.
- DQN eval win rates are noisy at 60–100 games; trust the multi-eval trend and
  the internal-Elo curve, not single points.
- Agent is not yet competitive with the Heuristic / Minimax — expected; next step
  is the AlphaZero-style Policy+Value+MCTS upgrade (`docs/alphazero-plan.md`).

## Next action
1. Finish + document the self-play run (section 3 of `RESULTS.md`).
2. (Optional) hyperparameter tuning pass on the DQN (γ=1.0, longer schedule).
3. Future phase: implement `docs/alphazero-plan.md`.

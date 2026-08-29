# Progress

_Last updated: 2026-08-29_

## Current phase
Phases 1–10 complete and validated. Remaining work is the optional future
AlphaZero-style upgrade (`docs/alphazero-plan.md`).

## Current task
None in flight.

## Phase 10 — Web app (play / fine-tune / analysis)
- `scripts/serve.py` → zero-dependency web app (`http.server` + vanilla JS).
- **Play**: full game vs the bot; **Analysis**: Lichess-style eval graph +
  per-move Best/Inaccuracy/Mistake/Blunder labels + suggested moves.
- **Fine-tune from a game**: grades each bot move (bot win-prob regret + 1-ply
  positional check), builds DQN transitions with the game result + shaping
  (hard negative for blunders, positive for best), trains on an anchored replay
  buffer, and **rolls the update back** if win rate vs Random drops.
- `webapp/bot_service.py::OthelloBot` is the stable interface for external
  testing; `scripts/bot_cli.py` is a `genmove`/`eval` line protocol.
- Bundled bot: `models/othello_bot_v1.pt` (curriculum + self-play checkpoint).
- Docs: `docs/webapp.md`. Tests: `tests/webapp/` + serve/CLI smoke.

## Completed
- **Phase 0–5** — scaffolding; tested Othello engine; Random/Greedy/Heuristic/
  Minimax agents; evaluation framework (tournaments, Wilson CI, internal Elo,
  reports); RL env; masked Double-DQN (network/replay/agent/trainer).
- **Phase 3 baseline eval** — `experiments/20260828-194501_eval/`.
- **Phase 6 — DQN curriculum, VALIDATED** —
  `experiments/20260828-201918_dqn_curriculum/` (380k steps):
  vs Random 0.55→0.82, vs Greedy 0.41→0.82, vs Heuristic 0.05→0.22; internal Elo
  1585→~1910. Reproducible gain vs Random on 2 seeds
  (`20260829-002434_repro_s777`: 0.27→0.71). Training deterministic given a seed.
- **Phase 7 — self-play, VALIDATED** —
  `experiments/20260829-012038_dqn_selfplay/` (120k steps from the curriculum
  checkpoint): vs Greedy 0.74→0.90, vs Heuristic 0.18→0.36; beats its own earliest
  snapshot 0.86 and mid snapshot 0.69; no forgetting (vs Random stays ~0.9).
- **Phase 8** — `scripts/track.py` + tracking artifacts committed under the
  curriculum run's `tracking/`.
- **Phase 9** — `scripts/play.py`.
- **Progress bars** — tqdm bar for `train.py` / `selfplay.py` (`--progress`).

All results: `experiments/RESULTS.md`.

## Test status
`python3 -m pytest` → 134 passed.

## Known issues
- Python 3.9.6 only; deps `--user`; run `python3 -m pytest`.
- CPU-only; a full curriculum run is ~3–4 h. DQN eval win rates noisy at 60–100
  games — trust multi-eval trends and the internal-Elo curve.
- Agent beats Random / Greedy decisively but is still ~0.35 vs the hand-written
  Heuristic and untested vs Minimax; closing that gap is the AlphaZero phase.

## Next action (optional / future)
1. Hyperparameter pass on the DQN (γ=1.0, longer ε schedule, bigger net).
2. Implement `docs/alphazero-plan.md` (Policy+Value net, MCTS, gated self-play).

# Progress

_Last updated: 2026-08-31_

## Current phase
Phases 1–11 complete and validated. Full spec audit done (see AUDIT below):
5 latent issues fixed with regressions; 2 spec-gap features added
(`scripts/eval_bot.py` standard protocol incl. Minimax; resumable self-play).
Phase 11: persistent model & checkpoint management.

## Phase 13 — Egaroucid for Console evaluation (2026-08-31)

`scripts/play_egaroucid.py` plays the production model against a locally-built
**Egaroucid for Console 7.8.1** (macOS ARM64) over **GTP** on stdin/stdout.
Bridge in `othello_rl/eval_external/` (`EgaroucidEngine` + `play_game`/`run_match`);
our own engine referees legality / passing / termination, Egaroucid is asked only
for its own moves (`genmove`). Model loaded once, `select_action` only (one masked
forward pass — no search). Results in `results/egaroucid/`.

- Baseline: `v001_curriculum_selfplay` **0 / 10** vs Egaroucid level 10
  (colours alternated, 4 random opening plies) — avg disc diff **−60**, and 0/6
  even at level 0. Consistent with the known ~0.08 vs Minimax-2.
- **Speed (Step 8 answer):** RL inference **~0.5 ms/move** (max < 2 ms), <0.15 s
  total for 10 games. The historical "5–6 min game" was the manual Othello Quest
  workflow / the web app's 3–5-ply *analysis* search — never the model, and that
  path is not used here.
- Model architecture / weights / training data / config / production checkpoint
  unchanged; only `eval_external/`, the script, `tests/eval_external/` (18) and
  `results/egaroucid/` were added. Docs: `docs/egaroucid-eval.md`.

## Current task
Phase 12 — historical-game supervised pretraining pipeline (see
`docs/historical-training.md`, plan in `.claude/plans/`). **12.1–12.6 all done**
(ingest → validate → analyze → datasets → supervised pretrain → eval/promote/
iterate). `PolicyValueNet` + DQN coexist behind `rl/checkpoint.py::load_agent`.
`experiments/index.jsonl` = committed append-only run log. Remaining: run the
pipeline on real WThor data; the RL half (AZ-MCTS self-play) is deferred to
`docs/alphazero-plan.md`. NB: the 3–5 ply analysis is a heuristic labeller, not an
oracle; `Board.apply` auto-skips forced passes; negamax @5 plies ≈ 4 pos/s/core.

## Model lifecycle (verified 2026-08-31)

The trained model is independent of the web-app runtime:

- **Weights** live in `checkpoints/` (+ curated `models/`); the **active**
  production model is named in `checkpoints/registry.json`.
- `scripts/serve.py` resolves the model from the registry, **verifies** it plays
  a legal opening move, prints its version/lineage, and never builds a random
  net (missing file → exit 2; initial-model fallback → loud warning).
- `OthelloBot` version/parent ride in the checkpoint meta, so a restart after a
  kept fine-tune keeps the version (was hardcoded to 0 before).
- In-app fine-tune writes only `webapp_state/` (scratch). Production changes
  **only** via `scripts/promote_model.py` (evaluate vs best/random/greedy/
  heuristic/minimax:2 → documented criterion → registry + `models/MODELS.md`).
- `format: 2` checkpoints carry optimizer + counters + RNG; `train.py --resume`
  continues training without resetting progress
  (`tests/rl/test_checkpoint.py`). Replay buffer re-warms (documented in
  `docs/training-and-models.md`); epsilon is `f(env_steps)`, no scheduler object.
- Rebuilding the frontend (`npm run build` → `web/dist`) touches no checkpoint
  (test: `test_appstate_does_not_touch_checkpoints`).

---

## AUDIT (2026-08-29)

Full codebase re-audit against `PROJECT_SPEC.md`. The repository was inspected
file by file; the engine and RL env were re-verified with fresh adversarial
checks (not just the existing tests).

### Subsystem status

| Subsystem | Status | Evidence |
|---|---|---|
| Othello game engine | **Verified** | 128 integration games across all baseline agent pairs with per-ply delta / flip-count invariants (0 failures); dihedral (rotation+reflection) equivariance of legal moves and flip sets; adversarial `flips_for_move` inputs (off-board, occupied, no-bracket, numpy ints); consecutive-pass / full-board / all-one-colour terminal detection; 200-game fuzz. |
| Baseline agents | **Verified** | All 4 agents always return legal moves & handle pass states (integration sweep); Greedy max-flips + deterministic tie-break; Heuristic perspective antisymmetry; **Minimax alpha-beta value == full unpruned negamax value** (5 seeds); minimax takes an immediate win. |
| Evaluation system | **Verified** | Reproducibility (identical `MatchResult` for a fixed seed); colour alternation; Wilson CI width vs n; internal Elo orders A>B>C and anchors correctly; report JSON/MD/PNG. Fixed: Elo now reads the recorded per-game colour instead of re-deriving it. |
| RL environment | **Verified** | Canonical `(3,8,8)` obs equals `encode_observation` for the learner at every step (150 episodes, both colours, 3 opponents); reward sign correct vs a referee (0 mismatches); sparse (0 mid-game); `terminated`/`truncated` never truncates a real game. |
| RL agent (DQN) | **Verified** | Masked greedy / eps-greedy only picks legal actions; `masked_q` sentinel; checkpoint round-trip (params + meta); `clone_network` is a frozen independent copy; forced-pass handled. |
| Training pipeline | **Verified** | Trainer smoke (finite loss, buffer fills, episodes advance); **bit-for-bit deterministic given a seed**; target-sync; eps schedule monotone; curriculum smoke writes metrics + checkpoints + resumes across stages. Empirically: Phase-6 curriculum learns (see Completed). |
| Self-play | **Verified** | Pool fallback / configurable distribution / recent-capacity / historical promotion; frozen snapshot independence; anti-forgetting eval; self-play smoke. Empirically validated (see Completed). |
| Checkpoint system | **Verified** | `DQNAgent.save/load/from_checkpoint` round-trip; forward-compatible `net_config` (missing `norm` key defaults); curriculum keeps per-stage + per-N-step checkpoints (never overwrites); bot fine-tune keeps `webapp_state/history/vNNNN.pt`. |
| Evaluation / metrics | **Verified** | `evaluate_agent`, `flatten_eval`, `tracking.track_checkpoints` (per-checkpoint win rate + round-robin internal Elo + curves), `MetricLogger` JSONL/CSV, experiment metadata (git commit + versions). |

Nothing was found that is "implemented but incorrect". The issues fixed in this
pass (below) were latent footguns / robustness gaps, not active miscomputation —
the empirical training results in `experiments/` stand.

### Fixed in the audit pass
- **`Board.__init__` aliased/froze the caller's array** (`np.asarray` on an
  already-int8 array returns it unchanged, then `writeable=False` mutated the
  caller's object). Now always takes an owned copy; internal transitions use a
  fast `_own=True` path so the hot loop is unaffected. Regression tests added.
- **`OthelloEnv.MAX_STEPS` was 80** — an Othello game can reach 120 plies
  (60 placements + up to 60 interleaved passes); a pathological long game would
  have been silently truncated with reward 0. Raised to 130; test asserts
  `>= 120` and that 120 random episodes never truncate.
- **`ratings_from_matches` re-derived each game's colour from `g_idx % 2`**
  instead of the actual assignment — silently wrong for
  `alternate_colors=False`. `GameResult` now records `a_is_black` / `a_score()`;
  Elo reads it (with a fallback for old results). Regression test added.
- **`FixedOpponentEnv.reset`** could return a terminal state if a very long
  random opening finished the game, tripping the next-step assertion. Now
  retries. Test with `opening_plies=58` added.
- Dead code removed (`_rollout` `pending`), misleading `nan_to_num` in the
  trainer replaced with an explicit `dones` mask.
- Removed an orphan/aborted committed run dir
  (`experiments/20260828-234251_dqn_curriculum`, baseline eval only).

## Phase 10 — Web app (play / fine-tune / analysis)
- Python JSON API (`scripts/serve.py`, stdlib `http.server`) + a **React + Vite**
  front end in `web/` (`npm run dev` proxies `/api` to the API; `npm run build`
  emits `web/dist` which the API server serves itself).
- **Play**: full game vs the bot; **Analysis**: Lichess-style eval graph +
  per-move Best/Inaccuracy/Mistake/Blunder labels + suggested moves +
  `?analyse=<transcript>` deep links.
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
`python3 -m pytest` → **248 passed** (~140 s). +18 this session for the
Egaroucid GTP bridge (`tests/eval_external/`, incl. a real-engine game that
skips when the executable is absent).

Extra verification run this pass (ad-hoc scripts, not in the suite): all-agent-pair
integration sweep with per-ply invariants; dihedral symmetry; RL-env reward-sign
referee check over 150 episodes — all clean.

---

# Known Issues

## Critical
- _(none)_

## Bugs
- _(none open — the five latent issues found in the audit are fixed and have
  regression tests; see the AUDIT section)_

## Missing Test Coverage
- [ ] `evaluation/report.py` `write_plot` / `write_markdown` exercised only via
      `test_report.py::test_generate_report_writes_files`; no assertion on the
      markdown table contents.
- [ ] `webapp/bot_service.finetune_from_game` not asserted to be *reproducible*
      (it is, by construction: seeded buffer + seeded sampling).
- [ ] No test drives `scripts/track.py` on real multi-stage checkpoints (only a
      2-checkpoint smoke).

## Technical Debt
- [x] ~~`OpponentPool.snapshot_state()` dead~~ → replaced with `save()`/`load()`;
      `run_self_play(resume_pool=...)` and `scripts/selfplay.py --resume <run>`
      restore the pool so a warm-started run continues the historical ladder.
- [ ] `webapp/moves.py::replay_positions` is now only used by its own test
      (`analyse_line` supersedes it). Harmless helper; keep or remove.
- [ ] `MinimaxAgent.search_value` doesn't reset `self.nodes` (diagnostic only).
- [ ] `bot_service._train` recreates the Adam optimizer every call — intentional
      (each in-app fine-tune is an independent short nudge on the *scratch* model;
      its optimizer state is deliberately not persisted). The *curriculum* trainer
      now does persist optimizer state — see `rl/checkpoint.py` + `train.py --resume`.
- [ ] `configs/*.yaml` and dataclass defaults duplicate hyperparameters; the
      config is the source of truth for scripts but tests use dataclass defaults.

## Potential Risks
- CPU-only + pure-Python engine: a full curriculum run is ~3–4 h. Sized configs
  accordingly; not a correctness risk.
- DQN eval win rates are noisy at 60–100 games/eval — always read multi-eval
  trends and the internal-Elo curve, never a single point.
- The trained agent beats Random/Greedy decisively (~0.92) but is ~0.31–0.35 vs
  the Heuristic / Minimax-1 and **0.08 / 0.03 vs Minimax-2 / -3** (now measured —
  `experiments/20260830-183834_eval_othello_bot_v1/`, internal Elo ~2002, between
  greedy and heuristic). Closing the gap to search-based play needs the
  AlphaZero-style upgrade (`docs/alphazero-plan.md`), not more DQN tuning.
- Internal Elo is a crude iterative fit (not Bradley-Terry MLE) — fine for
  within-project ordering, must never be quoted as an external rating.

## Environment
- Python 3.9.6 only; deps installed `--user` into `~/Library/Python/3.9`; run
  tests with `python3 -m pytest` (console script may not be on PATH).

## Next action (optional / future)
1. Hyperparameter pass on the DQN (γ=1.0, longer ε schedule, bigger net) and an
   eval vs Minimax depths 1–3.
2. Implement `docs/alphazero-plan.md` (Policy+Value net, MCTS, gated self-play).

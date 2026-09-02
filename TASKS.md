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

- [x] `agents/base.py` — `Agent` interface
- [x] `RandomAgent`
- [x] `GreedyAgent` (max immediate flips, deterministic tie-break)
- [x] `HeuristicAgent` (configurable weights)
- [x] `MinimaxAgent` (alpha-beta, configurable depth)
- [x] Tests: all agents always return legal moves
- [x] Tests: agents handle pass-only states
- [x] Tests: deterministic agents reproducible
- [x] Tests: minimax picks obvious win / corner; alpha-beta value == plain minimax

## Phase 3 — Baseline evaluation system

- [x] `evaluation/tournament.py` — play_match / round-robin, alternating colours, seeds
- [x] `evaluation/metrics.py` — W/L/D, mean score diff, Wilson CI
- [x] `evaluation/elo.py` — internal Elo (labelled "internal")
- [x] Machine-readable results (JSON)
- [x] `scripts/evaluate.py` + `configs/evaluation.yaml`
- [x] Performance report generator (markdown + plot)
- [x] Tests for tournament/metrics/elo

## Phase 4 — RL environment

- [x] `environment/environment.py` — `OthelloEnv` gym-like API
- [x] Canonical `(3,8,8)` observation + length-65 action mask
- [x] Sparse reward, done, info
- [x] Seeding
- [x] Tests: observation, mask, reward sign, termination, illegal action

## Phase 5 — Initial deep-RL agent

- [x] `docs/rl-algorithm.md` — justify DQN
- [x] `rl/network.py` — small conv net, Q-head length 65
- [x] `rl/replay_buffer.py`
- [x] `rl/agent.py` — masked epsilon-greedy / greedy eval, save/load
- [x] `rl/trainer.py` — DQN loop, target net, metrics
- [x] Tests: network shapes, masking, buffer, checkpoint round-trip, trainer smoke

## Phase 6 — Training vs fixed opponents

- [x] `rl/opponents.py` fixed-opponent env wrapper
- [x] `scripts/train.py` + `configs/train.yaml`
- [x] Stage 1: vs Random — reproducible win-rate gain over untrained net
- [x] Periodic eval + checkpointing + metrics JSON + plot
- [x] Stage 2: vs Random+Greedy
- [x] Stage 3: vs Heuristic
- [x] Document results

## Phase 7 — Self-play

- [x] `rl/self_play.py` — opponent pool, configurable sampling
- [x] Checkpoint pool management (`pool.save/load`; `pool.pt` beside checkpoints)
- [x] Anti-forgetting eval vs historical checkpoints
- [x] Resume a self-play run (`run_self_play(resume_pool=)` / `selfplay.py --resume`)
- [x] Tests: pool sampling distribution, snapshot independence, save/load round-trip, resume

## Phase 8 — Evaluation & experiment tracking

- [x] Standard eval protocol vs Random/Greedy/Heuristic/Minimax/historical
      — `evaluation/bot_report.py` + `scripts/eval_bot.py` (single checkpoint vs
      the full panel incl. Minimax 1/2/3, Wilson CIs, internal-Elo placement);
      `scripts/track.py` for the multi-checkpoint / historical view
- [x] `utils/logging.py` metric logger (JSONL + CSV)
- [x] Plots: games vs win-rate, iterations vs baseline perf, checkpoint vs Elo
- [x] Experiment metadata capture

## Phase 9 — Human play interface

- [x] `scripts/play.py` terminal UI

## Future — AlphaZero-inspired (NOT started — see `docs/alphazero-plan.md`)

- [ ] `rl/az_network.py` — shared torso → (policy logits 65, value scalar)
- [ ] `rl/mcts.py` — PUCT search over `Board`, Dirichlet root noise, masked priors
- [ ] `rl/az_selfplay.py` — MCTS self-play, store `(obs, π, z)` triples
- [ ] `rl/az_trainer.py` — `(z−v)² − πᵀ log p` loss over a self-play replay
- [ ] `rl/az_arena.py` — gate candidate vs current best (promote if > ~55%)
- [ ] `scripts/az_train.py` — self-play → train → gate iteration loop
- [ ] Milestone: iterated training shows rising internal Elo vs baselines + prev best

## Phase 10 — Web app (play · fine-tune · analysis)

- [x] `webapp/bot_service.py` — `OthelloBot` (stable tested interface): move
      selection, position/move evaluation, `analyse_game`, `finetune_from_game`
- [x] Move grading: bot win-prob regret + 1-ply positional check -> Lichess-style label
- [x] Fine-tune: game transitions + shaping + anchored replay buffer + guardrail rollback
- [x] `webapp/session.py` in-memory human-vs-bot game
- [x] `webapp/moves.py` transcript / move-list parser + position replay
- [x] `webapp/server.py` zero-dependency JSON API (serves `web/dist` or a fallback page)
- [x] `web/` React + Vite front end: Board / BoardArea / PlayPanel / AnalysisPanel
- [x] Analysis = interactive board (Lichess style): play moves on the board,
      legal-move dots, live per-position eval + "bot likes" lines, dashed best
      move, ⏮◀▶⏭ / arrow-key / graph / move-list nav, take-back, `?analyse=` deep link
- [x] `bot_service.analyse_line` + `GET /api/eval` back the interactive board
- [x] `npm run dev` (Vite proxies `/api` → :8000), `npm run build`, `npm run dev:all`
- [x] `scripts/serve.py` + `configs/webapp.yaml`
- [x] `scripts/bot_cli.py` line protocol for external harnesses (`genmove` / `eval`)
- [x] `models/othello_bot_v1.pt` bundled bot + `docs/webapp.md` + `web/README.md`
- [x] Tests: bot_service, moves, HTTP API, SPA fallback, serve/CLI smoke
- [x] Durable game dataset: `data/games.jsonl` (committed, append-only,
      cross-restart dedup by move sequence); `games_path` in `configs/webapp.yaml`
- [x] Analysis polish: corner-aware move ranking (best/"bot likes" never concedes
      a corner), grading override (give-corner→Blunder, take-corner→Best/Excellent,
      big-loss→Blunder), "Best" reserved for the true best, per-ply best-move hint,
      eval bar fills with the leading colour, move list scrolls itself not the page,
      "Save to dataset" button + `POST /api/games`
- [x] Fine-tune shaping: hard-penalise corner-losing moves, reinforce corner takes

## Phase 11 — Persistent model & checkpoint management

- [x] `rl/checkpoint.py` — `format: 2` checkpoint (weights + optimizer + counters
      + RNG + config + metrics + version/parent), back-compatible `load_checkpoint`,
      `restore_training`, `resolve_checkpoint` (`latest`/`best`/`vNNN`/path/run-dir)
- [x] `checkpoints/{initial,experiments,production}/` tree + `checkpoints/registry.json`
      (the active-model pointer; committed except `experiments/`)
- [x] `DQNAgent.save(**extra)` non-breaking; `DQNTrainer` resume (`state_dict` /
      `load_resume_state`); `run_curriculum(resume_state=)` writes a full checkpoint
- [x] `scripts/train.py --resume {latest,best,vNNN,path,run_dir}` — weights +
      optimizer + counters + RNG, architecture from the checkpoint
- [x] `scripts/promote_model.py` — evaluate candidate vs {best, random, greedy,
      heuristic, minimax:2}; documented promotion criterion; only writer of
      `production/` + registry + `models/MODELS.md`
- [x] `scripts/serve.py` — resolve from registry, verify a legal opening move,
      loud warning on initial-model fallback, never a random net; `GET /api/model`
- [x] `OthelloBot`: version/parent survive a restart (ride in checkpoint meta);
      `reset_to_baseline` restores the true base checkpoint
- [x] `docs/training-and-models.md`, `models/MODELS.md`
- [x] Tests: `tests/rl/test_checkpoint.py` (save/load identical preds, restart
      loads trained ckpt, frontend rebuild doesn't touch ckpts, interrupted
      training resumes, production protected from unevaluated / worse candidates,
      resolver, no silent V0)
- [x] RL: potential-based corner-safety reward shaping (`rl/shaping.py`, opt-in
      via `configs/*.yaml` `shaping:` block); `tests/rl/test_shaping.py`
- [x] Analysis grading = chess.com Expected Points model (EP lost vs the best
      move; one `_expected_points` number drives grade + "bot likes" + eval)

## Phase 12 — Historical-game supervised pretraining pipeline

Plan: `docs/historical-training.md`. The 3–5 ply analysis is a heuristic
labeller, **not an oracle**.

- [x] 12.1 Ingestion — `ingest/` (`GameRecord`, source-pluggable parsers:
      WThor `.wtb`, our `games.jsonl`, transcripts, generic JSON), cross-source
      dedup by move signature, `scripts/ingest_games.py`, `docs/game-data-format.md`;
      `tests/ingest/test_ingest.py`
- [x] 12.2 Validation & replay — replay every placement through the engine,
      classify VALID/INVALID/INCOMPLETE/UNSUPPORTED, flag winner mismatch;
      `validation/`, `scripts/validate_games.py`, stats copied to `experiments/`;
      `tests/validation/test_validation.py` (9). (Engine auto-skips passes.)
- [x] 12.3 Short-horizon counterfactual analysis — `analysis/search.py` (shallow
      negamax + alpha-beta + exact-only transposition table, reuses `MinimaxAgent`
      ordering + `heuristic.evaluate`), `analysis/counterfactual.py`
      (regret = best_alt − played, `tanh` norm, BEST/GOOD/ACCEPTABLE/MISTAKE/
      BLUNDER), `analysis/pipeline.py` (+ process pool + `--benchmark`),
      `configs/analysis.yaml`, `scripts/analyze_games.py`,
      `docs/historical-training.md` (eval scale + oracle disclaimer + perf table);
      `tests/analysis/` (15): search==minimax, AB picks same move, TT exact,
      perspective, forced-pass path, regret arithmetic, horizon matters, sampling
      determinism, pipeline/benchmark
- [x] 12.4 Versioned datasets — `datasets/split.py` (game-level, pure hash of
      (seed, game_id), no leakage), `datasets/examples.py` (position -> obs +
      played move + outcome z + label, data_kind kept), `datasets/build.py`
      (strategies all / filtered / weighted; npz + manifest; leak assertion);
      `configs/dataset.yaml`, `scripts/build_dataset.py`; `tests/datasets/` (7)
- [x] 12.5 Policy(+value) net — `rl/az_network.py` (`PolicyValueNet`: shared torso
      -> 65 policy logits + tanh value), `rl/az_agent.py` (`PolicyValueAgent`,
      masked argmax, implements `Agent`), `rl/checkpoint.py` `net_kind` + `load_agent`
      dispatch (+ `save_policy_value_checkpoint`), `rl/supervised.py`
      (`SupervisedTrainer`: weighted CE policy + MSE value, per-epoch val loss +
      move accuracy, `.resume`), `scripts/pretrain.py`, `configs/pretrain.yaml`,
      `utils/experiment.py::log_experiment` -> `experiments/index.jsonl`.
      Behaviour cloning, NOT RL. `tests/rl/test_supervised.py` (6): overfit tiny
      set, val-only metrics, ckpt under experiments/ + roundtrip, resume restores
      epoch+optimizer, agent always legal + plays, DQN ckpt still loads.
- [x] 12.6 `scripts/eval_bot.py` loads any net kind (`load_agent`) + `--vs-production`
      / `--vs <ckpt>`; `scripts/promote_model.py --config` (promotion rule from
      `configs/pretrain.yaml`, + `min_games`); `experiments/index.jsonl` tracking
      (`utils/experiment.py::log_experiment` in pretrain/eval/promote/iterate);
      `scripts/iterate.py` (subprocess orchestrator, `--dry-run`/`--from`/`--to`),
      `configs/iterate.yaml`; `tests/rl/test_iterate_and_tracking.py` (7)
- [ ] (deferred) AZ-MCTS self-play — `docs/alphazero-plan.md` (the RL half of the loop)

## Phase 13 — External-engine evaluation: Egaroucid for Console

- [x] `othello_rl/eval_external/` — evaluation-only bridge to a local
      **Egaroucid for Console 7.8.1** (built from source, macOS ARM64):
      `egaroucid.py` (`EgaroucidEngine`: GTP subprocess over stdin/stdout —
      `clear_board` / `play` / `genmove` / `gogui-rules_final_result`; coord
      conversions; auto-discovery — `--egaroucid` / `$EGAROUCID_EXE` / repo-root
      `Egaroucid-console*/` (git-ignored) / `~/Downloads` — + build hint) and `match.py`
      (`play_game` / `run_match`: our engine referees legality/passing/
      termination, per-move RL inference timing, `assert move in legal_moves`,
      forced-pass logging + counting, replayable `transcript`, Egaroucid-vs-our
      -engine verdict cross-check).
- [x] `scripts/play_egaroucid.py` — loads the production model **once**
      (registry-resolved, never a random net), prints version + checkpoint,
      one debug game or an N-game colour-alternating match; writes
      `results/egaroucid/match_*.json` + `summary.json`.
- [x] `--train` (opt-in): fine-tune the model on the match games via the existing
      `OthelloBot.finetune_from_games` (behaviour cloning + shaping + guardrail
      rollback); `--train-loops N` for play→learn iteration; kept updates saved as
      a **candidate** under `checkpoints/experiments/egaroucid_ft_*.pt` (git-ignored)
      — production/registry untouched, promotion stays a separate script step.
      `eval_external.match.{records_to_training_games,finetune_on_records}`.
- [x] `scripts/train_vs_egaroucid.py` — unattended long run (`--hours 8`):
      play→fine-tune every round until a deadline, Egaroucid level ramps
      `--level-start`→`--level-end`. Storage-light (no match JSONs): only
      latest/best/hourly/final `.pt` + `progress.jsonl` + `run.json` under
      `checkpoints/experiments/egaroucid_train_<stamp>/`. `best.pt` picked by a
      periodic vs-Random+Greedy check; anchor buffer periodically refilled;
      engine auto-restarts on failure; `Ctrl-C`/`touch STOP` finalises; `--resume`.
      Ends with a base-vs-final-vs-best eval. Production/registry untouched.
- [x] `docs/egaroucid-eval.md` (protocol, build command, the exact run command,
      baseline result, the speed finding, the `--train` flow).
- [x] `tests/eval_external/test_egaroucid_bridge.py` (21): coord round-trips,
      GTP parser, fake-engine game loop (both colours), forced-pass logging,
      illegal-RL-move abort, colour alternation, real-engine game when the
      executable is present, `records_to_training_games`, `finetune_on_records`
      (guardrail keep/rollback ⇔ weights move/restore).
      Baseline: v001_curriculum_selfplay 0/10 vs Egaroucid level 10
      (avg −60 discs); RL inference ~0.5 ms/move.

## Phase 14 — search engine for the web app

- [x] `othello_rl/engine/bitboard.py` — bitboard move-gen / flips / conversion,
      property-tested equal to the numpy `environment` engine (legal moves,
      resulting positions, final score) over random games.
- [x] `othello_rl/engine/solver.py` — negamax + alpha-beta, TT, static+TT move
      ordering, iterative deepening under a time budget, **exact endgame solve**
      from `endgame_empties` (leaf = final disc margin). `best_move()` -> `(sq,
      value, meta{depth,exact,nodes,pv})`.
- [x] `OthelloBot.best_move()` + `engine_budget` / `engine_endgame`; wired into
      `session.bot_move`, `evaluate_position` (#1 move + winprob), `bar_eval`;
      `_ANALYSE_BUDGET` (0.12s) for the analysis graph; `POST /api/best_move`.
      `_DEFAULT_ENGINE_BUDGET` patched to 0 by `tests/webapp/conftest.py`.
- [x] `tests/engine/` (6): bitboard == numpy engine; exact endgame == brute-force
      negamax; iterative deepening; `best_move` payload; evaluate_position top ==
      engine; engine beats `MinimaxAgent(2)` >=5/6. Beats the old shallow
      suggestion 17-0-3. Model / training / RL env unchanged.
- [x] `scripts/train_from_engine.py` — behaviour-clone the engine into the DQN:
      generate engine self-play (explore for variety, engine move = label at
      every ply), train Q-head CE toward the engine move + value head toward the
      outcome, generate→train→eval loop for `--hours`. Candidate under
      `checkpoints/experiments/engine_bc_<stamp>/`; production untouched.
      `tests/engine/test_train_from_engine.py` (2).

# Historical-game supervised pretraining pipeline (Phase 12)

Turn permitted historical Othello games into a scientifically defensible,
reproducible training pipeline:

```
historical games (WThor .wtb, our games.jsonl, transcripts)
   ↓  scripts/ingest_games.py     — parse -> GameRecord JSONL (+ dedup)
   ↓  scripts/validate_games.py   — replay every move through the engine; VALID / INVALID / INCOMPLETE
   ↓  scripts/analyze_games.py    — 3–5 ply counterfactual search -> per-move quality label
   ↓  scripts/build_dataset.py    — versioned, game-level train/val/test split; all / filtered / weighted
   ↓  scripts/pretrain.py         — SUPERVISED imitation pretraining of a policy(+value) net (NOT RL)
   ↓  scripts/eval_bot.py         — tournament vs baselines + production + historical checkpoints
   ↓  scripts/promote_model.py    — promote only if the documented criterion is met
   ↺  scripts/iterate.py          — repeat
```

Data formats and ingestion: **`docs/game-data-format.md`**.

## Terminology

A **ply is one move by one player.** `lookahead_plies: [3, 5]` in
`configs/analysis.yaml` means 3 and 5 *individual moves* of look-ahead — not 3–5
full turns.

## Short-horizon move analysis (`scripts/analyze_games.py`)

For every reconstructed position, a shallow **negamax search with alpha-beta
pruning** (`src/othello_rl/analysis/search.py`, reusing the tournament
`MinimaxAgent` move-ordering and the `heuristic_agent.evaluate` leaf) estimates
the value of the played move and of every legal alternative, `lookahead_plies`
deep. Then:

```
regret       = best_alternative_value − played_move_value      # heuristic-eval units
regret_norm  = tanh(max(0, regret) / evaluation_scale)          # 0..1
label        = BEST / GOOD / ACCEPTABLE / MISTAKE / BLUNDER     # thresholds on regret_norm
```

### How position quality is calculated — the evaluation scale

The leaf is `heuristic_agent.evaluate(board, player, weights)`, a weighted sum of
interpretable features from the **player-to-move's perspective**:

| feature | normalised range | default weight |
|---|---|---|
| `disc_diff` (my − opp discs) | ~[−1, 1] | 1 |
| `mobility` (my − opp legal moves) | [−1, 1] | 8 |
| `corners` (my − opp corners) / 4 | [−1, 1] | 25 |
| `corner_danger` (my − opp X/C-squares by an empty corner) / 8 | [−1, 1] | −8 |
| `edges` (my − opp edge discs) / 24 | [−1, 1] | 4 |

So a non-terminal `evaluate` sits roughly in **[−40, +40]**, dominated by corners
and mobility; a terminal position returns ±(10^6 + disc margin). `regret` (a
*difference* of two such values) is typically **0–15** for a real mistake and
much larger when a move loses a corner. `evaluation_scale` (default **6.0**) maps
that onto `tanh`, so e.g. a 6-unit swing → `regret_norm ≈ 0.76` → BLUNDER.
Thresholds (`configs/analysis.yaml::move_quality`) are on `regret_norm`, all
configurable.

### This is a heuristic labeller, **not an Othello oracle**

- The leaf evaluation is a **static heuristic**, not a solver.
- The horizon is **tiny** (3–5 plies). Othello is deeply positional — a move that
  looks bad in 5 plies can be correct, and vice-versa.
- The opponent is modelled by the *same* shallow search, not real play.

The labels exist to **weight / filter training data** (12.4) and to flag likely
human errors — never treat them as ground truth, and never claim a historical
move is objectively bad on their basis alone.

### Performance

Pure-Python engine + negamax. Measured on this machine (~8 legal moves/position,
every alternative searched, exact-only transposition table):

| horizon | ~positions / sec (1 core) |
|---|--:|
| 3 plies | ~70 |
| 4 plies | ~17 |
| 5 plies | ~4 |

A full ~60-ply game at horizon 5 ≈ 15 s single-core. For a large corpus use
`n_workers` (games are independent — near-linear scaling) and/or
`max_alternatives` (search the played move plus a deterministic sample of N
alternatives). Always run `scripts/analyze_games.py --benchmark --limit 100`
first and size the run from the real numbers — never fabricate them.

Output: `data/processed/analyzed_games/<source>.jsonl`, one game per line, every
move judged at every horizon (`by_horizon`). Stats (label distribution,
positions/sec, config) → `experiments/<ts>_analyze_<source>/analysis.stats.json`.

## Later stages

12.4 datasets, 12.5 policy(+value) supervised pretraining, 12.6 evaluation /
promotion / iterate loop — added as those phases land. The RL half of the loop
(AZ-style MCTS self-play) is deferred to `docs/alphazero-plan.md`.

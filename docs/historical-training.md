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

## Versioned training datasets (`scripts/build_dataset.py`)

Analysed games -> `(position, played move, game outcome)` training examples.
**Never train equally on every move** — a strong human still errs. Three
configurable strategies (`configs/dataset.yaml::strategy`):

| strategy | effect |
|---|---|
| `all` | every valid move, weight 1 (the baseline) |
| `filtered` | keep only `filtered_keep` labels (BEST/GOOD/ACCEPTABLE by default) |
| `weighted` | `label_weights` per label; BLUNDER → 0 (dropped) |

`horizon` picks which analysis horizon's labels to use.

**Splitting is at the game level** — `datasets/split.py::assign_split(game_id,
ratios, seed)` is a pure hash of `(seed, game_id)`, so a game (and *every* one of
its positions) is entirely in train, val or test. `build_dataset` asserts no
`game_id` appears in more than one split. Split assignment is stable as the
corpus grows.

Each `TrainingExample` carries `data_kind` ∈ {`historical`, `self_play`,
`engine_generated`} so historical and generated data stay distinguishable and can
be mixed with metadata later.

Output: `data/processed/training_data/<version>/{train,val,test}.npz`
(`obs (N,3,8,8) f32`, `policy (N,)`, `value (N,)` ∈ {−1,0,1} from the mover's
POV, `weight (N,)`, `label`, `data_kind`, `game_idx`, `game_ids`) + `manifest.json`
(config, per-split counts, label/data-kind histograms, mean weight; copied to
`experiments/<ts>_dataset_<version>/`). `<version>` = timestamp + config hash.

## Supervised pretraining (`scripts/pretrain.py`)

**This is imitation / behaviour cloning, NOT reinforcement learning.** The target
is a fixed `(position → move)` mapping (weighted cross-entropy on the policy head)
plus the game outcome `z` (MSE on the value head):

```
loss = weighted_CE(policy_logits, played_move) + value_loss_weight · MSE(value, z)
```

The network is a dedicated **`PolicyValueNet`** (`rl/az_network.py`): a shared conv
torso → 65-way policy logits + a scalar tanh value. It is a *different model kind*
from the DQN `SmallOthelloNet`; both coexist behind
`rl/checkpoint.py::load_agent(path)`, which reads `net_kind` and returns a
`DQNAgent` or a `PolicyValueAgent` (`rl/az_agent.py`, masked-argmax over the
policy — plugs straight into `play_match` / `eval_bot.py` / `promote_model.py`).

Each epoch reports **val loss** and **top-1 move accuracy** on the val split (a
different set of games from train — see 12.4). `--resume <ckpt>` restores the
epoch counter, optimizer and RNG. Output → `checkpoints/experiments/<v>_pretrain.pt`
(`net_kind: policy_value`, `method: supervised_pretrain`, `dataset_version`,
`parent`), an `experiments/<ts>_pretrain/` run dir, and one row in
`experiments/index.jsonl`. It **never touches** `checkpoints/production/`.

Loss numbers alone say nothing about strength — every candidate must still be
evaluated by playing games (12.6).

## Later stages

12.6 evaluation / promotion / iterate loop — added as it lands. The RL half of the
loop (AZ-style MCTS self-play) is deferred to `docs/alphazero-plan.md`.

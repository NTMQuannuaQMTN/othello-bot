# Training & model lifecycle

How the trained bot is produced, versioned, resumed, evaluated, promoted and
served — and why restarting or rebuilding the web app can never reset it.

## Five separated concerns

| # | concern | lives in | who writes it |
|---|---|---|---|
| 1 | architecture / code | `src/othello_rl/rl/` (`network.py`, `agent.py`) | source control |
| 2 | trained weights | `checkpoints/` + curated `models/` | `train.py`, `selfplay.py`, `promote_model.py` |
| 3 | training state (optimizer, counters, RNG) | *inside* each checkpoint (`format: 2`) | `save_checkpoint` |
| 4 | training data | `data/games.jsonl` (web games); replay buffers are transient | the web app (append-only) |
| 5 | frontend / UI / scratch model | `web/`, `webapp_state/` | Vite, the web app |

Editing/rebuilding/restarting the frontend touches only (5). `npm run build`
writes `web/dist` and nothing else. In-app "Fine-tune" writes only
`webapp_state/` (a **scratch** model). The production checkpoint and the registry
change **only** via `scripts/promote_model.py`.

## Directory layout

```
checkpoints/
├── initial/     v000_initial.pt          the "V0" anchor (committed)
├── experiments/ vNNN.pt                   training / candidate outputs (gitignored)
├── production/
│   ├── best.pt                            current production model (committed)
│   └── latest.pt                          most recently promoted (committed)
└── registry.json                          THE active-model pointer (committed)

models/
├── othello_bot_v1.pt                      curated copies of promoted models (committed)
├── MODELS.md                              promotion history table
└── README.md

data/games.jsonl                           durable game dataset (committed, append-only)
webapp_state/                              scratch: current.pt, history/, info.json (gitignored)
```

## Checkpoint format (`src/othello_rl/rl/checkpoint.py`)

`save_checkpoint(path, agent, optimizer=, train_step=, episode=, games_played=,
train_config=, seed=, rng_state=, experiment=, metrics=, version=, parent=)`
writes a superset of the old `format: 1` dict, so:

- every old `.pt` (e.g. `models/othello_bot_v1.pt`) still loads unchanged;
- `load_checkpoint(path) -> Checkpoint` fills missing fields with sane defaults;
- `restore_training(path, agent, optimizer) -> ResumeState` loads weights +
  optimizer state and returns the counters/RNG for the trainer to adopt. The
  **replay buffer is not serialized** — it re-warms from fresh rollouts. There is
  no LR/epsilon scheduler object: epsilon is a pure function of `env_steps`
  (`DQNConfig.epsilon`), so it resumes correctly for free.

## Lifecycle

```
                data/games.jsonl
                      │
   ┌──────────────────┼─────────────────────────┐
   ▼                  ▼                          ▼
scripts/train.py   scripts/selfplay.py    scripts/finetune_from_games.py
 (curriculum)       (opponent pool)        (offline nudge from web games)
   │                  │                          │
   └────────►  checkpoints/experiments/vNNN.pt  ◄┘
                      │
                      ▼
              scripts/eval_bot.py            (vs the standard panel, Wilson CIs, internal Elo)
                      │
                      ▼
              scripts/promote_model.py <candidate>
                      │   evaluates candidate vs {current best, random, greedy, heuristic, minimax:2}
                      │   promotion criterion (see models/MODELS.md)
                      ▼
              checkpoints/production/best.pt + latest.pt
              checkpoints/registry.json
              models/othello_bot_<version>.pt   +  MODELS.md row
                      │
                      ▼
              scripts/serve.py   → loads registry.active_checkpoint_path()
```

### Train

```bash
python3 scripts/train.py --config configs/train.yaml            # fresh curriculum
python3 scripts/train.py --resume latest --steps-scale 1.0      # continue the production line
python3 scripts/train.py --resume experiments/<run_dir>         # continue a specific run
python3 scripts/train.py --resume checkpoints/experiments/v003.pt
```

`--resume` restores model weights, Adam optimizer state, `env_steps` /
`train_steps` / `episodes`, and RNG state; the architecture is taken from the
checkpoint (never the yaml) so weights and net config can't drift apart. It then
runs the configured stages again, with the counters continuing upward — stopping
and restarting **never** resets progress (verified in
`tests/rl/test_checkpoint.py::test_interrupted_training_resumes`).

Every run writes a full `format: 2` checkpoint to
`<run>/checkpoints/final.pt`. Training never writes to `checkpoints/production/`.

### Evaluate

```bash
python3 scripts/eval_bot.py --checkpoint checkpoints/experiments/v003.pt --games 200
```

### Promote (the only path to production)

```bash
python3 scripts/promote_model.py checkpoints/experiments/v003.pt \
    --name v003_selfplay --parent v001_curriculum_selfplay \
    --method "self-play from v001" --games 200
```

Pass → copies the candidate to `checkpoints/production/{best,latest}.pt`,
rewrites `registry.json`, copies to `models/othello_bot_v003_selfplay.pt`, adds a
`MODELS.md` row, writes `experiments/<ts>_promote_v003/promotion.json`.
Fail → prints the comparison table, exits 1, writes nothing.

### Serve

`scripts/serve.py` resolves the model in this order:
`--checkpoint` → `configs/webapp.yaml: checkpoint` (if set) →
`checkpoints/registry.json` → `checkpoints/initial/v000_initial.pt` →
`models/othello_bot_v1.pt`. It then **verifies** the model plays a legal opening
move and prints its version/lineage. It never constructs a random network; a
missing file is a hard error (exit 2). If it falls back to the initial model
(no registry) it prints a loud warning. `GET /api/model` reports what's loaded.

## Storing checkpoints in git

`SmallOthelloNet` checkpoints are ~1.6 MB — fine to commit directly, which is why
`checkpoints/production/` and `models/` are tracked. If a future architecture
pushes checkpoints past ~10 MB, move `checkpoints/production/*.pt` and
`models/*.pt` to **Git LFS** (or attach them to GitHub releases and have
`promote_model.py` download them); the registry/pointer scheme is unchanged.
`checkpoints/experiments/*.pt` always stay local.

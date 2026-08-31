# Model registry — promotion history

The **active** production model is recorded in
[`../checkpoints/registry.json`](../checkpoints/registry.json); the backend
(`scripts/serve.py`) loads whatever that file points at. This table is the
human-readable log of every promotion. Only `scripts/promote_model.py` writes
here or to `checkpoints/production/`.

Win rates are vs 200 games/opponent, random 4-ply openings, colour-alternated
(draw = 0.5). "vs best" = vs the *previous* production model.

| version | date | parent | method | vs Random | vs Greedy | vs Heuristic | vs Minimax:2 | vs prev-best | notes |
|---|---|---|---|--:|--:|--:|--:|--:|---|
| v000_initial | 2026-08-29 | – | adopted baseline | – | – | – | – | – | random-init `SmallOthelloNet` (2×32ch), the "V0" anchor |
| v001_curriculum_selfplay | 2026-08-29 | v000_initial | 380k-step fixed-opponent curriculum + 120k-step self-play | 0.93 | 0.90 | 0.36 | 0.08 | – | `othello_bot_v1.pt`; internal Elo ~2002 (`experiments/20260830-183834_eval_othello_bot_v1`) |

## Promotion criterion

`scripts/promote_model.py <candidate>` promotes **only** if all hold:

1. `wilson_lb(win_rate vs current best) > 0.50` — beats the incumbent with 95% confidence.
2. `win_rate vs Random >= (best's vs Random) - 0.03` — no regression vs the weak baselines.
3. `win_rate vs Greedy  >= (best's vs Greedy)  - 0.03`.

Otherwise nothing is written and the script exits non-zero. `--force` overrides
(recorded as `"forced": true` in the registry).

See [`../docs/training-and-models.md`](../docs/training-and-models.md) for the full lifecycle.

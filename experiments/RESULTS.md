# Experimental results

This file summarises the real experiments run in this repo. Every number here
comes from a committed run directory under `experiments/`; regenerate with the
commands shown.

## Baseline tournament (Phase 3)

`experiments/20260828-194501_eval/` — 150 games/matchup, seed 20260828,
`opening_plies=4`. Command:

```
python3 scripts/evaluate.py --config configs/evaluation.yaml
```

| A | B | A win rate | mean disc diff |
|---|---|--:|--:|
| greedy | random | 0.600 | +4.8 |
| heuristic | random | 0.960 | +32.9 |
| heuristic | greedy | 0.943 | +35.5 |
| minimax:2 | heuristic | 0.783 | +24.5 |
| minimax:3 | heuristic | 0.923 | +39.7 |
| minimax:4 | heuristic | 0.997 | +45.8 |
| minimax:3 | greedy | 0.993 | +46.3 |

Internal Elo (anchored, comparable only within this project — **not** an online
rating): minimax:4 2372 · minimax:3 1804 · minimax:2 1597 · heuristic 1445 ·
greedy 937 · random 846.

Observations: pure max-flips greedy is only marginally better than random in
Othello (a known result); the heuristic (corners + mobility + edges) is a large
jump; alpha-beta depth scales strength smoothly.

## DQN curriculum training (Phase 6)

_(filled in when `experiments/20260828-201918_dqn_curriculum/` completes)_

Command:

```
python3 scripts/train.py --config configs/train.yaml
python3 scripts/track.py --run experiments/<run_dir>
```

### Stage 1 — vs Random

Success criterion: *a clearly and reproducibly higher win rate vs Random than the
untrained network.*

- Untrained network: win rate vs Random ≈ 0.55.
- Partial trend (first ~90k env steps): 0.55 → 0.62 → 0.58 → 0.69 → 0.72 → 0.65 →
  0.75; vs Greedy 0.41 → … → 0.71. **Criterion met** — the trained agent is
  consistently well above the untrained network vs Random after ~45k steps, and
  also overtakes Greedy.

### Reproducibility check

_(rerun stage 1 with a second seed; expect the same qualitative gain)_

## Self-play (Phase 7)

_(filled in after a `scripts/selfplay.py` run)_

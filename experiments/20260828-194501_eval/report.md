# Baseline evaluation report

_Generated 2026-08-28T11:55:12.485806+00:00 · seed 20260828_

## Matches

| A | B | games | A wins | B wins | draws | A win rate | 95% CI | mean disc diff | sig.? |
|---|---|------:|------:|------:|-----:|----------:|:------:|--------------:|:---:|
| greedy | random | 150 | 87 | 57 | 6 | 0.600 | [0.52, 0.67] | +4.78 | yes |
| heuristic | random | 150 | 143 | 5 | 2 | 0.960 | [0.92, 0.98] | +32.91 | yes |
| heuristic | greedy | 150 | 141 | 8 | 1 | 0.943 | [0.89, 0.97] | +35.49 | yes |
| minimax:2 | heuristic | 150 | 117 | 32 | 1 | 0.783 | [0.71, 0.84] | +24.46 | yes |
| minimax:3 | heuristic | 150 | 138 | 11 | 1 | 0.923 | [0.87, 0.96] | +39.69 | yes |
| minimax:4 | heuristic | 150 | 149 | 0 | 1 | 0.997 | [0.97, 1.00] | +45.80 | yes |
| minimax:3 | greedy | 150 | 149 | 1 | 0 | 0.993 | [0.96, 1.00] | +46.31 | yes |

## Internal Elo (internal)

> Internal experimental Elo — comparable only within this project. Not an external/online rating.

| agent | rating |
|---|---:|
| minimax:4 | 2372 |
| minimax:3 | 1804 |
| minimax:2 | 1597 |
| heuristic | 1445 |
| greedy | 937 |
| random | 846 |

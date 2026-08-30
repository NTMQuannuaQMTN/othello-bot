# Standard evaluation — othello_bot_v1

_2026-08-30T10:38:34+00:00 · 120 games/opponent · seed 20260829 · random openings_

| opponent | W–L–D | win rate | 95% CI | mean disc diff | sig.? |
|---|---|--:|:--:|--:|:--:|
| random | 111–8–1 | 0.929 | [0.87, 0.96] | +19.9 | yes |
| greedy | 107–10–3 | 0.904 | [0.84, 0.94] | +20.1 | yes |
| heuristic | 39–76–5 | 0.346 | [0.27, 0.43] | -11.3 | yes |
| minimax:1 | 35–81–4 | 0.308 | [0.23, 0.40] | -12.7 | yes |
| minimax:2 | 9–110–1 | 0.079 | [0.04, 0.14] | -31.5 | yes |
| minimax:3 | 3–115–2 | 0.033 | [0.01, 0.08] | -38.3 | yes |

## Internal Elo (internal, random = 1500)

> Comparable only within this panel. Not an external rating.

| agent | rating |
|---|--:|
| minimax:3 | 2599 |
| minimax:2 | 2450 |
| heuristic | 2138 |
| minimax:1 | 2111 |
| othello_bot_v1 **(this bot)** | 2002 |
| greedy | 1650 |
| random | 1500 |

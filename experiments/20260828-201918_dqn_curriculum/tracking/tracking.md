# Checkpoint strength tracking

Internal Elo (internal) — comparable only within this run.

| checkpoint | env_steps | vs random | vs greedy | vs heuristic | internal Elo |
|---|--:|--:|--:|--:|--:|
| untrained | 0 | 0.550 | 0.450 | 0.033 | 1585 |
| stage1_random_step30000 | 30000 | 0.642 | 0.525 | 0.067 | 1695 |
| stage1_random_step60000 | 60000 | 0.733 | 0.550 | 0.100 | 1622 |
| stage1_random_step90000 | 90000 | 0.800 | 0.692 | 0.083 | 1840 |
| stage1_random_step120000 | 120000 | 0.725 | 0.700 | 0.067 | 1839 |
| stage2_random_greedy_step165000 | 165000 | 0.692 | 0.683 | 0.200 | 1765 |
| stage2_random_greedy_step225000 | 225000 | 0.925 | 0.792 | 0.325 | 1920 |
| stage3_heuristic_step285000 | 285000 | 0.917 | 0.833 | 0.308 | 1905 |
| stage3_heuristic_step345000 | 345000 | 0.858 | 0.892 | 0.175 | 1874 |
| stage3_heuristic_step375000 | 375000 | 0.850 | 0.800 | 0.217 | 1910 |

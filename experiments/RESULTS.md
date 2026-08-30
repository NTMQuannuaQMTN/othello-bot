# Experimental results

Every number here comes from a committed run directory under `experiments/`.
Reproduce with the commands shown. Internal Elo is **project-internal only** — it
is anchored within a single evaluation pool and is not comparable to any online
Othello rating.

---

## 1. Baseline tournament (Phase 3)

`experiments/20260828-194501_eval/` — 150 games/matchup, seed 20260828,
`opening_plies=4`.

```
python3 scripts/evaluate.py --config configs/evaluation.yaml
```

| A | B | A win rate | 95% CI | mean disc diff |
|---|---|--:|:--:|--:|
| greedy | random | 0.600 | [0.52, 0.67] | +4.8 |
| heuristic | random | 0.960 | [0.92, 0.98] | +32.9 |
| heuristic | greedy | 0.943 | [0.89, 0.97] | +35.5 |
| minimax:2 | heuristic | 0.783 | [0.71, 0.84] | +24.5 |
| minimax:3 | heuristic | 0.923 | [0.87, 0.96] | +39.7 |
| minimax:4 | heuristic | 0.997 | [0.97, 1.00] | +45.8 |
| minimax:3 | greedy | 0.993 | [0.96, 1.00] | +46.3 |

Internal Elo: `minimax:4 2372 · minimax:3 1804 · minimax:2 1597 · heuristic 1445 ·
greedy 937 · random 846`.

Notes: pure max-flips **greedy is only marginally better than random** in Othello
(a known result — flipping the most discs early surrenders mobility); the
positional **heuristic** (corners, mobility, edges, corner-danger) is a large
jump; alpha-beta depth scales strength smoothly.

---

## 2. DQN curriculum training (Phase 6)

`experiments/20260828-201918_dqn_curriculum/` — masked Double-DQN, 3-stage
curriculum, **380 000 env steps**, ~3.5 h on CPU, seed 20260828.
Net: 2 conv blocks × 32 ch, no norm, ~110k params. `configs/train.yaml`.

```
python3 scripts/train.py  --config configs/train.yaml
python3 scripts/track.py  --run experiments/20260828-201918_dqn_curriculum
```

### Success criterion (Stage 1 — vs Random)

> *A clearly and reproducibly higher win rate vs Random than the untrained network.*

**Met.** Two independent seeds:

| run | untrained vs Random | trained vs Random (stage-1 end) |
|---|--:|--:|
| seed 20260828 (full curriculum) | 0.55 | **0.72 – 0.89** (last 4 evals 0.72 / 0.89 / 0.70 — then rises further in later stages) |
| seed 777 (`20260829-002434_repro_s777`, stage 1 only) | 0.27 | **0.71** (peak 0.78) |

Training is deterministic given a seed (unit test
`test_training_is_deterministic_given_seed`; identical network parameters on
repeated runs).

### Strength across the whole run (`scripts/track.py`, 60 games/baseline, seed 4242)

| checkpoint | env steps | vs random | vs greedy | vs heuristic | internal Elo |
|---|--:|--:|--:|--:|--:|
| untrained | 0 | 0.55 | 0.45 | 0.03 | 1585 |
| stage1 | 30k | 0.64 | 0.53 | 0.07 | 1695 |
| stage1 | 60k | 0.73 | 0.55 | 0.10 | 1622 |
| stage1 | 90k | 0.80 | 0.69 | 0.08 | 1840 |
| stage1 | 120k | 0.72 | 0.70 | 0.07 | 1839 |
| stage2 (random+greedy) | 165k | 0.69 | 0.68 | 0.20 | 1765 |
| stage2 | 225k | 0.93 | 0.79 | 0.33 | 1920 |
| stage3 (+heuristic) | 285k | 0.92 | 0.83 | 0.31 | 1905 |
| stage3 | 345k | 0.86 | 0.89 | 0.18 | 1874 |
| stage3 | 375k | 0.85 | 0.80 | 0.22 | 1910 |

`final.pt` end-of-training eval (100 games, `opening_plies=4`):
**Random 0.815 · Greedy 0.815 · Heuristic 0.215**, up from the untrained
**0.55 / 0.41 / 0.05**.

Plots: `winrate_vs_steps.png`, `train_return.png`,
`tracking/winrate_vs_checkpoint.png`, `tracking/elo_vs_checkpoint.png`.

### Reading the result

- The agent goes from ~coin-flip to **decisively beating Random and Greedy** and
  reaches ~0.2–0.33 vs the strong hand-written Heuristic (which it only meets in
  stage 3). Milestone 1 (beat Random reproducibly) and milestone 2 (meaningful
  performance vs Greedy) are both achieved.
- Eval curves are **noisy at 60–100 games/eval** (±~0.06 at p≈0.8); the trend
  over several evals and the internal-Elo curve (1585 → ~1910) are the reliable
  signal, not single points. The dip at 345k is within noise.
- Not yet competitive with the Heuristic or Minimax — expected for a small CPU
  DQN with sparse rewards and ~10k games of experience. The path forward is the
  AlphaZero-style upgrade (`docs/alphazero-plan.md`).

---

## 3. Self-play with opponent pool (Phase 7)

`experiments/20260829-012038_dqn_selfplay/` — warm-started from the curriculum
`final.pt`, **120 000 self-play env steps**, opponent pool
`{baseline 0.2, historical 0.3, recent 0.5}`, snapshot every 12k steps.
`configs/selfplay.yaml` (`--steps-scale 0.6`).

```
python3 scripts/selfplay.py --config configs/selfplay.yaml \
    --init experiments/20260828-201918_dqn_curriculum/checkpoints/final.pt
```

| point | vs Random | vs Greedy | vs Heuristic | vs `hist0` (earliest snapshot) | vs `hist_mid` |
|---|--:|--:|--:|--:|--:|
| start (curriculum `final.pt`) | 0.895 | 0.735 | 0.18 | 0.44 | – |
| 60k steps | 0.905 | 0.875 | 0.285 | 0.70 | – |
| **end (120k steps)** | **0.925** | **0.895** | **0.36** | **0.86** | **0.69** |

Plot: `selfplay_winrate.png`.

### Reading the result

- Self-play **kept improving the agent** past where the fixed-opponent curriculum
  left it: vs Greedy 0.74 → 0.90, vs Heuristic 0.18 → 0.36.
- **Newer agents beat older versions** — the central self-play question. The
  current agent scores 0.86 vs its earliest snapshot and 0.69 vs a mid-training
  snapshot (both started near 0.5).
- **No catastrophic forgetting**: win rate vs Random stays 0.89–0.93 for the whole
  run, because 20 % of self-play games are still against the fixed baselines and
  30 % against historical snapshots.

---

## 4. Standard evaluation of the bundled bot (Phase 8 protocol)

`experiments/20260830-183834_eval_othello_bot_v1/` — `models/othello_bot_v1.pt`
(the curriculum + self-play checkpoint), 120 games/opponent, random openings,
seed 20260829.

```
python3 scripts/eval_bot.py --checkpoint models/othello_bot_v1.pt --games 120
```

| opponent | W–L–D | win rate | 95% CI | mean disc diff |
|---|---|--:|:--:|--:|
| random | 111–8–1 | **0.929** | [0.87, 0.96] | +19.9 |
| greedy | 107–10–3 | **0.904** | [0.84, 0.94] | +20.1 |
| heuristic | 39–76–5 | 0.346 | [0.27, 0.43] | −11.3 |
| minimax:1 | 35–81–4 | 0.308 | [0.23, 0.40] | −12.7 |
| minimax:2 | 9–110–1 | 0.079 | [0.04, 0.14] | −31.5 |
| minimax:3 | 3–115–2 | 0.033 | [0.01, 0.08] | −38.3 |

Internal Elo (this panel, `random` = 1500): `minimax:3` 2599 · `minimax:2` 2450 ·
`heuristic` 2138 · `minimax:1` 2111 · **`othello_bot_v1` 2002** · `greedy` 1650 ·
`random` 1500.

### Reading the result

- The bot **decisively beats Random and Greedy** (~0.9, CIs well clear of 0.5)
  and sits ~1 ply of search below the hand-written Heuristic / Minimax-1
  (~0.31–0.35). It is **clearly outclassed by 2–3-ply alpha-beta search**
  (0.08 / 0.03) — this is the first measurement of the bot vs Minimax, and it
  confirms the expected ceiling for a ~410k-param CPU DQN with a purely sparse
  reward.
- Closing the gap to search-based play is exactly what the AlphaZero-style
  upgrade (`docs/alphazero-plan.md`) is for; more DQN tuning will not get there.

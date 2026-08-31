# Playing the RL bot against Egaroucid for Console

`scripts/play_egaroucid.py` runs the **trained** Othello RL bot against a local
**Egaroucid for Console** engine over GTP and records the games. It is an
*evaluation-only* integration — it never trains, fine-tunes, promotes, or writes
to `models/` / `checkpoints/`.

## TL;DR — the command

```bash
export PATH="$HOME/Library/Python/3.9/bin:$PATH"   # this machine's Python

# one debug game, every move printed, RL bot as Black
python3 scripts/play_egaroucid.py

# a 10-game mini-match (colours alternate), results saved under results/egaroucid/
python3 scripts/play_egaroucid.py --games 10

# pin things explicitly
python3 scripts/play_egaroucid.py \
    --checkpoint checkpoints/production/best.pt \
    --games 10 --level 10 \
    --egaroucid ~/Downloads/Egaroucid-console_v7.8.1/bin/Egaroucid_for_Console.out
```

Key flags: `--games N`, `--level 0-60` (Egaroucid strength, default 10),
`--opening-plies K` (random opening plies for game diversity, default 4),
`--threads`, `--nobook`, `--start-color black|white`, `--seed`, `--quiet`,
`--out-dir`, `--no-save`, `--move-timeout`.

With `--checkpoint` omitted the bot is resolved from `checkpoints/registry.json`
(the active production model — currently `v001_curriculum_selfplay`,
`checkpoints/production/best.pt`). A random network is never constructed.

## The Egaroucid engine

* **Executable:** built from source at
  `~/Downloads/Egaroucid-console_v7.8.1/bin/Egaroucid_for_Console.out`
  (auto-discovered; override with `--egaroucid`).
* **Version:** Egaroucid for Console 7.8.1, macOS ARM64 (Generic build, Clang).
* **OS compatibility:** Windows ships prebuilt; macOS / Linux must be built from
  source. This machine is Apple Silicon (M-series), so it was built with:

  ```bash
  cd ~/Downloads/Egaroucid-console_v7.8.1
  xattr -cr .
  clang++ -O2 ./src/Egaroucid_for_Console.cpp -o ./bin/Egaroucid_for_Console.out \
      -mtune=native -pthread -std=c++20 -DHAS_NO_AVX2 -DHAS_ARM_PROCESSOR
  ```

  (`-DHAS_NO_AVX2 -DHAS_ARM_PROCESSOR` = the Generic/ARM path; the default SIMD
  build needs x86 AVX2.) It must be run from its own `bin/` directory so it finds
  `resources/eval.egev2`, `resources/book.egbk3`, etc. — the wrapper does this.

### Protocol: GTP over stdin/stdout

Egaroucid speaks the **Go Text Protocol** when started with `-gtp`
(`src/console/gtp_command*.hpp`). We use a tiny subset:

| command | use |
|---|---|
| `clear_board` | reset to the Othello start position, Black to move |
| `play <colour> <sq>` | tell Egaroucid a move (`black`/`white`; `<sq>` = `D3` or `pass`) |
| `genmove <colour>` | Egaroucid picks & plays that colour's move; replies `F5` or `PASS` |
| `gogui-rules_final_result` | Black-normalised verdict string (cross-check only) |
| `name` / `version` / `protocol_version` | identification |

Coordinates: files `A`–`H` = columns 0–7, ranks `1`–`8` = rows 0–7 — identical
to this project's `(row, col)` after subtracting 1 from the rank, so GTP `D3` ==
project square `"d3"` == `(row=2, col=3)`.

Passes: Egaroucid auto-passes internally whenever it is asked to move for the
side that is *not* to move, so an explicit "pass" message is never required —
our engine and Egaroucid stay in sync from the real moves alone.

> Note: Egaroucid's `final_score` command reports B/W from the side-to-move's
> perspective (not Black-normalised) and is unreliable as a verdict — we use
> `gogui-rules_final_result` and, above all, our own engine.

## How the bridge works

```
Python RL bot  ──select_action(Board)──►  our engine (othello_rl.environment)
                                              │  = referee: legality, passing,
                                              │    termination, final score
                                              ▼
                            othello_rl.eval_external.match.play_game
                                              │  play <our move> / genmove
                                              ▼
                            othello_rl.eval_external.egaroucid.EgaroucidEngine
                                              │  GTP on stdin/stdout
                                              ▼
                                   Egaroucid_for_Console.out -gtp
```

* **Our `Board` / `rules` are the single source of truth.** Egaroucid is asked
  only for *its* moves. Every RL move is checked with `assert move in
  legal_moves` before it is played; an illegal move stops the game and dumps the
  board, the legal moves and the model's top-Q output (it has never happened —
  `select_action` already masks to legal moves).
* The model is loaded **once** at start-up (`OthelloBot.load`), `.eval()`, kept
  in memory for the whole match. Start-up prints the version + checkpoint.
* Random opening plies (`--opening-plies`, default 4) are played by both sides so
  a match between two deterministic engines still yields distinct games.
* Forced passes are detected (our `Board.apply` auto-skips a blocked side),
  logged as `Move N: … → PASS`, and counted (`n_passes`).

Code: `src/othello_rl/eval_external/{egaroucid,match}.py`,
tests `tests/eval_external/test_egaroucid_bridge.py`.

## Results

Written to `results/egaroucid/`:

* `match_YYYYMMDD_HHMMSS.json` — one file per run: model version + checkpoint,
  Egaroucid version/config, per-game moves (SAN + GTP + action index +
  per-move RL inference time), passes, winner, final score, a replayable
  `transcript` (real placements only), and the aggregate summary.
* `summary.json` — an appended one-line-per-match history.

### Baseline run (2026-08-31)

`v001_curriculum_selfplay` (`checkpoints/production/best.pt`, 409,954-param
`SmallOthelloNet`, 32ch×2 blocks) vs **Egaroucid 7.8.1 level 10**, 10 games,
alternating colours, 4 random opening plies, seed 20260831:

| metric | value |
|---|--:|
| games | 10 |
| RL wins / losses / draws | 0 / 10 / 0 |
| win rate | 0.0% |
| avg RL discs | 0.6 |
| avg disc diff (RL − Egaroucid) | −60.0 |
| RL inference — mean / median / max | 0.5 / 0.5 / 1.6 ms |
| total RL thinking time | 0.11 s |
| total tournament wall time | ~14 s |
| Egaroucid verdict agreed with our engine | 10 / 10 |

The RL bot is crushed — expected: it already scores only ~0.08 vs a depth-2
minimax (`PROGRESS.md`), and Egaroucid is one of the strongest Othello engines in
the world even at low levels. At `--level 0` (1-ply) the margin narrows to about
−36 discs but it is still 0/6.

### Speed finding (Step 8)

RL inference is **~0.5 ms/move** (max < 2 ms), total < 0.15 s for a 10-game
match. The old "5–6 minute game" was **not** the model — it was the manual
Othello Quest workflow / the expensive 3–5-ply analysis path used by the web
app's *analysis* board. `play_egaroucid.py` uses only `OthelloBot.select_action`
(a single masked forward pass), never that search.

## What is guaranteed unchanged

Model architecture, weights, training data, training config and the production
checkpoint are untouched. Only `src/othello_rl/eval_external/`,
`scripts/play_egaroucid.py`, `tests/eval_external/` and `results/egaroucid/` were
added.

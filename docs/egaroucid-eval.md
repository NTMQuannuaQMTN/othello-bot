# Playing the RL bot against Egaroucid for Console

`scripts/play_egaroucid.py` runs the **trained** Othello RL bot against a local
**Egaroucid for Console** engine over GTP and records the games. By default it is
pure evaluation — the model is never modified. With `--train` it *also*
fine-tunes the model on the games it just played and writes the result as a
**candidate** under `checkpoints/experiments/`; the production checkpoint and
`checkpoints/registry.json` are still never touched (see
[Learning from the match](#learning-from-the-match)).

## TL;DR — the command

Run it **from the repo root** (or give the script an absolute path):

```bash
cd /Users/qnrj/Code/othello-bot        # or use an absolute path to the script

# one debug game, every move printed, RL bot as Black
python3 scripts/play_egaroucid.py

# a 10-game mini-match (colours alternate), results saved under results/egaroucid/
python3 scripts/play_egaroucid.py --games 10

# pin things explicitly
python3 scripts/play_egaroucid.py \
    --checkpoint checkpoints/production/best.pt \
    --games 10 --level 10 \
    --egaroucid Egaroucid-console_v7.8.1/bin/Egaroucid_for_Console.out
```

The script is directory-independent (it resolves the repo root from its own
path); only the *shell* needs to find `scripts/play_egaroucid.py`, so `cd` to the
root **or** give an absolute path — `python3 scripts/play_egaroucid.py` from
`web/` fails because there is no `web/scripts/`.

Either `python3` (3.9.6) on this machine has PyTorch (the `--user` site-packages
under `~/Library/Python/3.9` are shared); if yours does not, use `/usr/bin/python3`.

Key flags: `--games N`, `--level 0-60` (Egaroucid strength, default 10),
`--opening-plies K` (random opening plies for game diversity, default 4),
`--threads`, `--nobook`, `--start-color black|white`, `--seed`, `--quiet`,
`--out-dir`, `--no-save`, `--move-timeout`. Training: `--train`,
`--train-loops N`, `--train-grad-steps`, `--train-lr`, `--train-guardrail-games`,
`--train-out`.

With `--checkpoint` omitted the bot is resolved from `checkpoints/registry.json`
(the active production model — currently `v001_curriculum_selfplay`,
`checkpoints/production/best.pt`). A random network is never constructed.

## The Egaroucid engine

* **Executable:** `Egaroucid-console_v7.8.1/bin/Egaroucid_for_Console.out`
  (built from source). The engine folder lives in the **repo root** and is
  **git-ignored** (`/Egaroucid-console*/` in `.gitignore`) — it is large and not
  ours. `find_egaroucid()` looks, in order: `--egaroucid` / `$EGAROUCID_EXE`
  (a file *or* a folder), `$PATH`, `<repo>/Egaroucid-console_v7.8.1/bin/`,
  `~/Downloads/Egaroucid-console*/bin/`, then a shallow `Egaroucid-console*/bin/`
  glob of the repo root and `~/Downloads`.
* **Version:** Egaroucid for Console 7.8.1, macOS ARM64 (Generic build, Clang).
* **OS compatibility:** Windows ships prebuilt; macOS / Linux must be built from
  source. This machine is Apple Silicon (M-series), so it was built with:

  ```bash
  cd Egaroucid-console_v7.8.1
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

## Learning from the match

`--train` fine-tunes the model on the games it just played, then writes the
result as a candidate. It reuses the project's existing behaviour-cloning path
(`OthelloBot.finetune_from_games`), nothing new:

1. Each game teaches the bot **its own** side's moves (real placements only; the
   replay re-inserts forced passes).
2. DQN transitions carry the game outcome (a loss vs Egaroucid → −1) plus the
   usual conservative shaping (hard-penalise conceding a corner, reinforce a
   corner take / an unambiguous best move).
3. One training pass on an **anchored** replay buffer (the bot's own play vs
   Random/Greedy is mixed in so 10 games can't overwrite the policy).
4. **Guardrail:** win rate vs Random is measured before and after; if it drops by
   more than `guardrail_margin` (0.10) the update is **rolled back** and nothing
   is written.

```bash
python3 scripts/play_egaroucid.py --games 10 --train
python3 scripts/play_egaroucid.py --games 6 --train --train-loops 3   # play→learn→play→…
```

A kept update is saved to `checkpoints/experiments/egaroucid_ft_<stamp>.pt`
(git-ignored, like every training candidate). It is **not** promoted — evaluate
and promote it exactly like any other candidate:

```bash
python3 scripts/eval_bot.py     --checkpoint checkpoints/experiments/egaroucid_ft_<stamp>.pt --vs-production
python3 scripts/promote_model.py checkpoints/experiments/egaroucid_ft_<stamp>.pt   # only if it passes
```

Example (10 games vs level 10, seed 20260831): fine-tune #1 — 90 reinforced /
11 penalised, TD loss 0.077 → 0.052, win% vs Random 0.933 → 0.900 (within the
guardrail) → kept. The match JSON gains a `"training"` block with the per-round
`FineTuneReport`. Learning from 10 one-sided losses is not expected to close the
gap to Egaroucid — it is exposed because the pipeline supports it and the
guardrail makes it safe.

## What is guaranteed unchanged

Without `--train`, the model is untouched. **With** `--train` the in-memory model
is fine-tuned and a candidate `.pt` is written under `checkpoints/experiments/` —
but model **architecture**, the **training dataset**, the **training config**,
`checkpoints/production/` and `checkpoints/registry.json` are never modified;
promotion stays a deliberate, separate `scripts/promote_model.py` step. Added:
`src/othello_rl/eval_external/`, `scripts/play_egaroucid.py`,
`tests/eval_external/`, `results/egaroucid/`, `.gitignore` lines for the engine
folder + `.DS_Store`, and this doc.

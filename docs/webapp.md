# Web app — play, fine-tune, analyse

**Backend** = Python JSON API (`http.server`, no extra deps) — bot inference,
analysis, fine-tuning. **Frontend** = React + Vite in `web/`.

```bash
# dev (hot reload) — needs BOTH the API (:8000) and Vite (:5173)
cd web && npm install && npm run dev:all      # runs both; open http://localhost:5173

# ...or in two terminals:
python3 scripts/serve.py --config configs/webapp.yaml     # terminal 1  -> API on :8000
cd web && npm run dev                                     # terminal 2  -> :5173 (proxies /api)

# production (single process)
cd web && npm run build          # -> web/dist
python3 scripts/serve.py         # serves web/dist on http://127.0.0.1:8000
```

Notes:
- Open **`http://localhost:5173`** (the URL Vite prints), not `127.0.0.1:5173`
  — Vite may bind IPv6-only.
- `scripts/serve.py` resolves its config paths against the repo root, so
  `npm run api` / `npm run dev:all` work even though npm runs it from `web/`.
- If the API isn't up yet the app shows a red banner with the command to run and
  recovers automatically once it responds.
- With no `web/dist` build, `scripts/serve.py` still runs the API and serves a
  short "how to build the front end" page at `/`.

## Play tab

- Start by choosing your colour — **Black / White / Random** (Black moves first).
  `?play=black` (or `white` / `random`) skips the chooser.
- Legal squares are dotted; your / the bot's last move is outlined. The bot
  waits ~0.5 s after your move before replying (`/api/move` with
  `bot_reply:false`, then `/api/bot_move`).
- **Move history** (same list + `⏮ ◀ ▶ ⏭` / arrow-key navigation as the Analysis
  tab): click any move to view the board at that point; the list scrolls to keep
  the current move visible; "▶ back to live position" returns. The current game
  survives a page refresh.
- When the game ends: **Analyse this game** (opens the Analysis board on it),
  **Fine-tune from this game**, and **Fine-tune from all N saved games** (see
  below).

## Analysis tab (Lichess-analysis style)

An **interactive analysis board** — you don't type moves, you play them:

- legal moves are shown as dots; click a square to add it to the line;
- **`bot likes: c4 70% · f5 70% · …`** lists the bot's top moves for the current
  position (click one to play it);
- the **best move you can play now is always outlined with a dashed box** (and
  named in the legend); the move you actually played is outlined in blue and
  graded (Best / Excellent / Good / Inaccuracy `?!` / Mistake `?` / Blunder `??`).
  The move list shows the best alternative for **every** ply, not just mistakes;
- navigate with ⏮ ◀ ▶ ⏭, the arrow keys, or by clicking the eval graph / move
  list; **take back** (or Backspace) pops the last move; playing a move while
  viewing an earlier position replaces the continuation from there;
- **use last game** loads the game you just played; **paste game** accepts a
  move list (`f5 d6 c3 …`) or run-together transcript (`f5d6c3…`);
- a `?analyse=<transcript>` URL opens straight into that line.

The **eval bar** and **eval graph** are a win-probability for **Black** across the
line, from the fast positional score (disc diff, mobility, corners, edges, corner
danger) from Black's fixed perspective — *not* the DQN value, which is
side-to-move-relative and just produces a per-ply zig-zag. It is sharpened as the
board fills so a decided endgame reads as ~0 / 1. The bar fills from the bottom
with Black's share, so the colour that's ahead fills the bar.

Below the graph, a **strategy read-out** per side: move-quality counts +
`accuracy` (fraction of Good-or-better moves), corners / X-squares / edge moves
played, average mobility, disc count.

**Save to dataset** appends the current line to `data/games.jsonl` so you can
batch-train on many games later (the Play tab auto-saves every finished game).

**Teach the bot this game** — *Learn the whole game* (both sides) / *⚫ Black
only* / *⚪ White only*. `learn_color` = `"both"` / `"black"` / `"white"`; each
side is graded and reinforced from its own perspective, outcome-weighted. Point
it at a strong player's game to reinforce that side's Best/Excellent moves
(`POST /api/finetune {moves, learn_color}`). The Play tab's game-over screen has
the same *Learn the whole game* / *Learn the bot's moves* choice, plus *Learn
from all N saved games* (`scripts/finetune_from_games.py --learn both` offline).

### How a move is graded

Grading follows chess.com's **Expected Points** model. Every legal move gets an
**expected-points** value `EP(move)` = the mover's win probability after playing
it (1 = winning, 0.5 = even, 0 = losing), from `bot_service.py::_expected_points`:

```
EP(move) = 0.8 · positional_winprob(after move, mover's view)   # disc/mobility/edge/corner heuristic
         + 0.2 · bot_win_prob(move)                             # the DQN's (noisy) opinion
         −       corner_penalty(move)                           # see below
```

A move is graded by **expected points lost** = `EP(best move) − EP(played move)`:

| classification | expected points lost |
|---|---|
| Best | 0.00 |
| Excellent | (0, 0.02] |
| Good | (0.02, 0.05] |
| Inaccuracy `?!` | (0.05, 0.10] |
| Mistake `?` | (0.10, 0.20] |
| Blunder `??` | (0.20, 1.00] |

The **dashed best move**, the **"bot likes"** list and `GET /api/eval`
(`moves[].winprob` = EP, `moves[].ep_lost`, `moves[].corner_risk`) are all just
`EP`, sorted. So the move shown as best has 0 expected points lost → always grades
**Best**; it can never come back as `?!`. A move that is not the #1 pick is never
"Best". Win probabilities for the played move and the best move are shown in the
move list, the board legend and the status line.

**Corner penalty** (folded straight into `EP`, so an X-square really does show
fewer expected points). An X/C-square move (b2/g2/b7/g7 etc.) is only penalised
when the opponent can *actually* force the corner — checked with a short
"can the opponent take this corner within one more exchange, even if I defend?"
search (`_corner_forcible`) — not merely because the square is an X-square:

```
opponent can play straight into a corner now              -> −0.32 EP
X-square move, opponent can force the corner              -> −0.24 EP
C-square move, opponent can force the corner              -> −0.11 EP
X-square move but the corner stays safe                   -> −0.04 EP  (loose only)
move takes a corner                                       -> +0.06 EP
```

Fine-tuning is conservative about which moves it reinforces: the game outcome is
always the base signal; an *extra* bonus/penalty is only added for taking vs
conceding a corner, a clear positional blunder the 1-ply check agrees with, or the
unambiguous best move in a game that side won.

## Fine-tuning the bot from a game

`POST /api/finetune` (the "Fine-tune" button) does:

1. Rebuild every position; grade each **bot** move with the rule above.
2. Build DQN transitions for the bot's moves with the **game result** as the
   terminal reward (`+1 / 0 / −1`).
3. **Shaping** — for a move graded Mistake/Blunder where a 1-ply check disagrees,
   add a hard negative transition for the played action and a positive one for
   the better move; for a clearly-best move, add a small positive.
4. Add those transitions (×`emphasis`) to a replay buffer that also holds
   ~2 000 **anchor** transitions of the bot playing Random/Greedy, so one game
   can't overwrite the policy.
5. Run `grad_steps` Double-DQN updates at a low LR.
6. **Guardrail** — play `guardrail_games` vs Random before and after. If the win
   rate dropped by more than `guardrail_margin`, the update is **rolled back**.
   Otherwise it's kept and versioned to `webapp_state/history/`.

All of this is configurable in `configs/webapp.yaml`. `POST /api/bot/reset`
restores the original weights.

### Using played games for future training

Every finished game is appended to **`data/games.jsonl`** (committed, append-only,
deduplicated by move sequence across restarts — path configurable via
`games_path` in `configs/webapp.yaml`), one line of
`{ts, human_color, moves, winner, score, bot_version}`. You can:

- **In the app** — "Fine-tune from all N saved games" on the game-over screen
  (`POST /api/finetune_all`): batches every saved game into one training pass.
- **Offline** —
  ```
  python3 scripts/finetune_from_games.py \
      --games data/games.jsonl --checkpoint checkpoints/production/best.pt \
      --out checkpoints/experiments/v002.pt --grad-steps 400
  ```
  Same guardrail (kept only if it doesn't weaken the bot vs Random). To make the
  result the model the site serves, promote it:
  `python3 scripts/promote_model.py checkpoints/experiments/v002.pt --name v002 --games 200`
  (see [`training-and-models.md`](training-and-models.md)).

The game log is plain JSONL — you can also feed it into your own training code
(`OthelloBot.finetune_from_games(list_of_game_dicts)` or roll your own using the
engine + `rl/` trainer).

### Which model is loaded

`GET /api/model` (alias `GET /api/bot`) reports the live bot: `version`, `parent`,
`source` (the resolved production checkpoint), `baseline` (whether
`reset_to_baseline` restores the true base checkpoint or just the loaded state),
`train_env_steps`, `games_finetuned`, and the `dataset` path + `dataset_games`
count. On startup `scripts/serve.py` prints the same and verifies the model plays
a legal opening move.

## The bot as a testable component

The bot is a stable, importable object:

```python
from othello_rl.webapp.bot_service import OthelloBot
from othello_rl.environment.board import Board

bot = OthelloBot.load("models/othello_bot_v1.pt")   # or webapp_state/current.pt
bot.select_move(Board.initial())      # -> (row, col) or None
bot.select_action(board)              # -> 0..63, or 64 for pass
bot.evaluate_position(board)          # -> {winprob_black, moves:[...], ...}
bot.analyse_game([actions...])        # -> [MoveAnalysis, ...]
```

`OthelloBot` also implements the project's `Agent` interface (via the wrapped
`DQNAgent`), so it plugs straight into the evaluation framework:

```python
from othello_rl.evaluation.tournament import play_match
play_match(OthelloBot.load("models/othello_bot_v1.pt").agent, "heuristic", num_games=200)
```

For a non-Python harness there's a line protocol:

```
printf 'genmove f5d6c3\nquit\n' | python3 scripts/bot_cli.py --checkpoint models/othello_bot_v1.pt
# -> f4
```

Commands: `genmove <transcript>`, `eval <transcript>`, `name`, `quit`.

## HTTP API

| method + path | body | returns |
|---|---|---|
| `GET /api/bot` · `GET /api/model` | | loaded-model info (version, parent, source checkpoint, params, games fine-tuned, dataset) |
| `GET /api/state` | | current game state |
| `POST /api/new` | `{human_color, level}` | new game state |
| `POST /api/move` | `{action, bot_reply?}` | state after your move (+ the bot's reply unless `bot_reply:false`) |
| `POST /api/bot_move` | `{}` | state after the bot moves (bot plays first / deferred reply) |
| `GET /api/eval` | | the bot's read of the current game position |
| `POST /api/analyse` | `{moves` \| `transcript` \| `history_actions}` | `positions[]` (each `.eval.moves[]` has `winprob`=EP, `ep_lost`, `corner_risk`), `plies[]` (`played_winprob`, `best_winprob`, `drop`=expected-points lost, `label`), `eval_graph`, `summary`, `strategy` |
| `POST /api/finetune` | `{}` or `{moves, learn_color}` | fine-tune from a game, learning `learn_color`'s moves (default: the bot's side) |
| `GET /api/games` | | `{count, path}` of saved games |
| `POST /api/games` | `{moves` \| `transcript, human_color?, learn_color?}` | append a game to `data/games.jsonl` (dedup by move sequence) → `{saved, count, reason?}` |
| `POST /api/finetune_all` | `{}` | fine-tune from every saved game at once |
| `POST /api/bot/reset` | | restores baseline weights |

# Web app — play & analyse

**Backend** = Python JSON API (`http.server`, no extra deps) — bot inference and
analysis, **no training**. **Frontend** = React + Vite in `web/`. Training is
offline (`scripts/train_*.py`) and promoted with `scripts/promote_model.py`;
the deploy build serves a torch-free numpy policy (see [`deploy.md`](deploy.md)).

```bash
# dev (hot reload) — needs BOTH the API (:8000) and Vite (:5173), two terminals:
python3 scripts/serve.py                  # terminal 1  -> API on :8000
cd web && npm install && npm run dev      # terminal 2  -> :5173 (proxies /api)

# production (single process)
cd web && npm run build          # -> web/dist
python3 scripts/serve.py         # serves web/dist on http://127.0.0.1:8000

# mirror the torch-free Vercel deploy
python3 scripts/serve.py --policy web/api/policy.npz
```

Notes:
- Open **`http://localhost:5173`** (the URL Vite prints), not `127.0.0.1:5173`
  — Vite may bind IPv6-only.
- If the API isn't up yet the app shows a banner and recovers once it responds.
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
- When the game ends: **Analyse this game** opens the Analysis board on it.

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

## The search engine

Move selection and the eval both run a **real Othello engine**
(`othello_rl/engine/`): a bitboard board, negamax + alpha-beta with a
transposition table and iterative deepening under a time budget, and an **exact
endgame solve** once ~12 squares remain (leaf = the true final disc margin, so
the last dozen moves are played perfectly). The DQN policy is only a tiebreak.
It beats a 2-ply minimax ~every game and the shallow heuristic suggestion
17-0-3.

- **Play tab**: the bot plays `OthelloBot.best_move` (`self.engine_budget`, ~1.0s
  + exact from 16 empties; `engine_budget <= 0` turns it off — the raw policy).
- **`POST /api/best_move`** `{history_actions, time_budget?}` → the strongest move
  for a position (default 3s, solves from 16 empties). This is the one to hit for
  "what should I actually play here."
- **Analysis board**: every position — the suggested move, the eval graph **and**
  each move's grade — comes from the same engine search (`_ANALYSE_BUDGET` ~1.0s,
  + exact from `_ANALYSE_ENDGAME` empties), for **whoever is on move**. The board
  highlight is the best move *for the side to move in the position shown*; after
  your move that is the opponent's best reply. Each position is searched once per
  analysis (`_bm_memo`), and the prefix cache keeps re-analysis to the new tip.

The **eval bar** and **eval graph** are a win-probability for **Black** from that
engine — exact once the game is close to solved, otherwise a squashed search
score. The bar fills from the bottom with Black's share. In the Play tab the bar
tracks history navigation (`GET /api/eval` = live, `POST /api/eval
{history_actions}` = that position).

In the **analysis graph** the bar is **grade-smoothed** (`_smoothed_eval_graph`,
`_EVAL_SWING_CAP`): a shallow search genuinely swings a lot on some plies (a
corner changes hands, mobility flips) even when the move played was the best
available — that swing is horizon noise, not information, since the pre-move eval
already assumed best play. So the bar may move at most ~0.05 on a **Best** move,
~0.13 on a **Good** one, and freely on a **Blunder**; it still chases the true
value, just at most that much per ply, so a genuine multi-ply shift catches up
over a few moves instead of jumping. Each graph point keeps `eval_black_raw`.

Below the graph, a **strategy read-out** per side: move-quality counts +
`accuracy` (fraction of Good-or-better moves), corners / X-squares / edge moves
played, average mobility, disc count.

### How a move is graded

Grading follows chess.com's **Expected Points** model. Every legal move gets an
**expected-points** value `EP(move)` = the mover's win probability after playing
it (1 = winning, 0.5 = even, 0 = losing), from `bot_service.py::_expected_points`
— a shallow **look-ahead** search from each candidate move, so "Mistake" / "Blunder"
means the move's *3-5-ply outcome* is worse than the best move's:

```
EP(move) = 0.8 · winprob( negamax(after move, ~2 plies) )       # look-ahead, heuristic leaf
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

### Which model is loaded

`GET /api/model` (alias `GET /api/bot`) reports the live bot: `version`,
`parent`, `source` (the resolved checkpoint / `web/api/policy.npz`),
`train_env_steps`, `params`, `network`. On startup `scripts/serve.py` prints the
same and verifies the model plays a legal opening move.

To change the model the site serves: train offline (`scripts/train_*.py` →
`checkpoints/experiments/…`), evaluate (`scripts/eval_bot.py`), promote
(`scripts/promote_model.py`), and for the deploy re-run
`scripts/export_policy.py` and commit `web/api/policy.npz`.

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

The API is **stateless** — the Play tab sends `{human_color, history_actions}`
with every request.

| method + path | body | returns |
|---|---|---|
| `GET /api/bot` · `GET /api/model` | | loaded-model info (version, parent, source, params, network, train_env_steps) |
| `POST /api/new` | `{human_color}` | fresh game state (bot moves first if it's Black) |
| `POST /api/state` | `{human_color, history_actions}` | game state for that history |
| `POST /api/move` | `{human_color, history_actions, action, bot_reply?}` | state after your move (+ the bot's reply unless `bot_reply:false`) |
| `POST /api/bot_move` | `{human_color, history_actions}` | state after the bot moves |
| `POST /api/eval` | `{history_actions}` | the eval bar for that position |
| `POST /api/best_move` | `{history_actions, time_budget?}` | the engine's strongest move for a position |
| `POST /api/analyse` | `{moves` \| `transcript` \| `history_actions}` | `positions[]` (each `.eval.moves[]` has `winprob`=EP, `ep_lost`, `corner_risk`), `plies[]` (`played_winprob`, `best_winprob`, `drop`=expected-points lost, `label`), `eval_graph`, `summary`, `strategy` |

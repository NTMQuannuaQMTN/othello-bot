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

Play a full game against the bot (choose your colour). Legal squares are dotted;
your last move / the bot's last move is outlined. When the game ends you can hit
**Fine-tune bot from this game** (see below).

## Analysis tab (Lichess-analysis style)

An **interactive analysis board** — you don't type moves, you play them:

- legal moves are shown as dots; click a square to add it to the line;
- **`bot likes: c4 70% · f5 70% · …`** lists the bot's top moves for the current
  position (click one to play it);
- the bot's best move is outlined with a dashed box; after a move is played it's
  graded (Best / Excellent / Good / Inaccuracy `?!` / Mistake `?` / Blunder `??`);
- navigate with ⏮ ◀ ▶ ⏭, the arrow keys, or by clicking the eval graph / move
  list; **take back** (or Backspace) pops the last move; playing a move while
  viewing an earlier position replaces the continuation from there;
- **use last game** loads the game you just played; **paste game** accepts a
  move list (`f5 d6 c3 …`) or run-together transcript (`f5d6c3…`);
- a `?analyse=<transcript>` URL opens straight into that line.

The **eval graph** is the bot's win-probability for Black across the line (see the
note below on why it's blended with a positional score).

### How a move is graded

The DQN's raw action-values are only weakly separated, so a move's **regret** is a
blend of two signals:

```
regret = 0.5 · (win-prob the bot thinks the move gives up vs its own best move)
       + 0.5 · tanh( positional value lost vs a 1-ply corner/mobility/edge check / 18 )
```

The 1-ply positional check is the same heuristic used by `HeuristicAgent`; it
catches tactical errors (e.g. giving up a corner) that the small DQN is blind to.
`regret` (0–1) is mapped to a label by the table in
`webapp/bot_service.py::_CLASS_TABLE`.

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
| `GET /api/bot` | | bot info (version, params, games fine-tuned) |
| `GET /api/state` | | current game state |
| `POST /api/new` | `{human_color, level}` | new game state |
| `POST /api/move` | `{action}` | state after your move + the bot's reply |
| `POST /api/bot_move` | | state after the bot moves (bot plays first) |
| `GET /api/eval` | | the bot's read of the current game position |
| `POST /api/analyse` | `{moves` \| `transcript` \| `history_actions}` | `positions[]` (grid + legal moves + eval per ply boundary), `plies[]` (move grades), `eval_graph`, `summary` |
| `POST /api/finetune` | `{}` or `{moves, human_color}` | fine-tune report + move grades |
| `POST /api/bot/reset` | | restores baseline weights |

# Web app — play, fine-tune, analyse

```
python3 scripts/serve.py --config configs/webapp.yaml
# open http://127.0.0.1:8000
```

Zero extra dependencies — the backend is Python's `http.server`, the frontend is
vanilla HTML/CSS/JS in `src/othello_rl/webapp/static/`.

## Play tab

Play a full game against the bot (choose your colour). Legal squares are dotted;
your last move / the bot's last move is outlined. When the game ends you can hit
**Fine-tune bot from this game** (see below).

## Analysis tab (Lichess-analysis style)

Paste a move list (`f5 d6 c3 …` or the run-together transcript `f5d6c3…`) or hit
**Use last game**. You get:

- an **eval graph** — the bot's win-probability for Black across the game
  (click it, or a move in the list, to jump to that position);
- the board at the selected position, with the played move outlined and the
  suggested move dashed;
- a **per-move list** with a Lichess-style label and glyph
  (Best / Excellent / Good / Inaccuracy `?!` / Mistake `?` / Blunder `??`) and a
  suggested alternative.

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
| `POST /api/analyse` | `{moves` \| `transcript` \| `history_actions}` | per-ply analysis + eval graph + positions |
| `POST /api/finetune` | `{}` or `{moves, human_color}` | fine-tune report + move grades |
| `POST /api/bot/reset` | | restores baseline weights |

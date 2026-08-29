# OthelloRL web front end (React + Vite)

## Dev

```bash
# terminal 1 — the Python JSON API (bot inference, analysis, fine-tuning)
python3 ../scripts/serve.py --config ../configs/webapp.yaml     # :8000

# terminal 2 — the React dev server (hot reload; proxies /api -> :8000)
npm install
npm run dev                                                     # :5173
```

Open the URL Vite prints (http://localhost:5173).

`npm run dev:all` runs both at once (via `concurrently`).

## Build

```bash
npm run build      # -> web/dist
python3 ../scripts/serve.py --config ../configs/webapp.yaml
open http://127.0.0.1:8000     # scripts/serve.py serves web/dist directly
```

## Layout

```
src/
  main.jsx          mount
  App.jsx           header, tab switch, footer, bot info
  api.js            fetch wrapper + small helpers
  styles.css
  components/
    Board.jsx       8x8 grid of discs
    BoardArea.jsx   board + eval bar + status line
    EvalBar/Graph   the win-probability bar and the game eval graph
    PlayPanel.jsx   play a game, then fine-tune the bot from it
    AnalysisPanel.jsx  interactive Lichess-style analysis board — play moves on
                       the board, the bot evaluates each position live
                       (?analyse=<transcript> deep link)
    BotBadge / FineTuneResult
```

All game logic and the model live in the Python backend; this app is a thin
client over the JSON API documented in `../docs/webapp.md`.

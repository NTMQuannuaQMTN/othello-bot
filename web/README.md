# OthelloRL web front end (React + Vite)

## Dev

The app needs **two processes**: the Python JSON API on `:8000` and the Vite
dev server on `:5173` (which proxies `/api` → `:8000`). The easiest way:

```bash
npm install
npm run dev:all      # runs the Python API + Vite together (via concurrently)
```

Then open the URL Vite prints — **`http://localhost:5173`** (Vite may bind IPv6
only, so `127.0.0.1:5173` can fail even though `localhost:5173` works).

Or run them in two terminals:

```bash
# terminal 1
python3 ../scripts/serve.py --config ../configs/webapp.yaml     # :8000

# terminal 2
npm run dev                                                     # :5173
```

If you open the app before the API is up you'll see a red banner with the
command to run; the page recovers on its own once the API responds.

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

# Deploying the web app (Vercel)

The web app deploys to Vercel as **one project**: the Vite front end as static
files, and the bot API as a **Python serverless function** that runs
*inference only* — no PyTorch.

## Why torch-free

Vercel functions cap at **250 MB unzipped**; PyTorch is ~1 GB. The production
DQN is tiny (two 3×3 conv layers + two linear heads, ~410 k params), so
`othello_rl.rl.numpy_policy.NumpyPolicy` runs its forward pass in a few lines of
numpy. The search engine (`othello_rl/engine/`) was already pure Python. So the
function's only dependency is `numpy` (~30 MB).

What the deploy **can** do: play a full game vs the bot, the analysis board
(best move + grades + eval graph), `/api/best_move`. What it **cannot**: fine-tune
or save games to a dataset — those need the full install and a writable disk;
the buttons are hidden and the endpoints return 400.

## Project setup

The Vercel project's **Root Directory is `web`** (Vercel auto-detects the Vite
app there). Everything the deploy needs lives under `web/`:

| path | role |
|---|---|
| `web/vercel.json` | build command, function limits, `/api/*` + SPA rewrites |
| `web/api/index.py` | the Python function — loads the bot once, delegates to `othello_rl.webapp.server.dispatch` |
| `web/api/policy.npz` | exported weights (committed; ~1.5 MB) |
| `web/api/requirements.txt` | `numpy` only |
| `web/package.json` `vercel-build` | `cp ../src/othello_rl web/api/othello_rl` then `vite build` — the whole repo is cloned, so `../src` is reachable at build time |
| `.vercelignore` (repo root) | keeps torch / checkpoints / data / `pyproject.toml` out of the upload |

`web/api/othello_rl/` is git-ignored — it's re-created by the build (and by
`scripts/export_policy.py` for local testing).

## One-time: export the model

Run with the full install whenever the production checkpoint changes:

```bash
python3 scripts/export_policy.py            # -> web/api/policy.npz (+ a torch-vs-numpy check)
git add web/api/policy.npz && git commit -m "web: refresh exported policy"
```

## Deploy

Connect the GitHub repo in the Vercel dashboard, **set Root Directory to `web`**,
and every push to `main` deploys. Or:

```bash
npm i -g vercel
cd web && vercel --prod
```

No environment variables are required (the front end calls the API same-origin
at `/api/*`).

### Separate API host instead

Set `VITE_API_BASE=https://your-api.example.com` as a Vercel build env var and
the front end will call that host instead of `/api`. Useful if you later run the
**full** API (torch, fine-tuning) on Fly.io / Render / Modal.

## Run the deploy build locally

```bash
python3 scripts/serve.py --policy web/api/policy.npz   # torch-free, mirrors Vercel
cd web && npm run build                                # then reload http://127.0.0.1:8000
```

The normal `scripts/serve.py --config configs/webapp.yaml` still runs the full
torch bot with fine-tuning for local development.

## Notes / limits

- **Cold start**: first request after idle loads numpy + the 1.5 MB npz (~1 s).
- **`maxDuration` 60 s** (`web/vercel.json`): `/api/best_move` (3 s) and building an
  analysis line move-by-move (~0.5 s/move) are fine; importing a full 60-move
  game to analyse from cold is ~30 s — under the limit but not instant.
- The API is **stateless** — the Play tab sends `{human_color, history_actions}`
  with every request, so it works without a persistent process.

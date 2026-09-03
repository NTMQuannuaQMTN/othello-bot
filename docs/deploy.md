# Deploying the web app (Vercel)

**One Vercel project.** It builds the Vite front end (static) and serves
`web/api/index.py` as a Python serverless function on the **same origin**, so the
front end calls `/api/*` with no env vars and no CORS. Inference only — no
PyTorch, no training (that's offline, `scripts/train_*.py`).

## Why torch-free

Vercel functions cap at **250 MB unzipped**; PyTorch is ~1 GB. The production DQN
is tiny (two 3×3 conv layers + two linear heads, ~410 k params), so
`othello_rl.rl.numpy_policy.NumpyPolicy` runs its forward pass in a few lines of
numpy; the search engine (`othello_rl/engine/`) was already pure Python. The
function's only dependency is `numpy`.

## Project settings (do this once)

In the Vercel project → **Settings → Build and Deployment**:

| setting | value |
|---|---|
| **Root Directory** | **`web`** |
| Framework Preset | Vite (auto-detected) |
| Build / Output / Install Command | leave as auto (or from `web/vercel.json`) |

Then **Deployments → Redeploy**. Every push to `main` deploys after that.

Do **not** run a separate "api" project — one project serves both.

## What's under `web/`

| path | role |
|---|---|
| `web/vercel.json` | function limits + `/api/*` and SPA rewrites |
| `web/api/index.py` | the function — loads the bot once, delegates to `othello_rl.webapp.server.dispatch` |
| `web/api/policy.npz` | exported weights (committed, ~1.5 MB) |
| `web/api/othello_rl/` | the package, **vendored + committed** (~300 KB) so the function bundle is self-contained |
| `web/api/requirements.txt` | `numpy` |
| `.vercelignore` (repo root) | keeps torch / checkpoints / data / `pyproject.toml` out of the upload |

## When the model changes

```bash
python3 scripts/export_policy.py     # -> web/api/policy.npz + re-vendors web/api/othello_rl
git add web/api/policy.npz web/api/othello_rl && git commit -m "web: refresh model"
```

## Run the deploy build locally

```bash
python3 scripts/serve.py --policy web/api/policy.npz   # torch-free, mirrors Vercel
cd web && npm run build                                # then reload http://127.0.0.1:8000
```

`python3 scripts/serve.py` (no `--policy`) runs the full torch bot from the
production checkpoint for local development.

## Notes / limits

- **Cold start**: first request after idle loads numpy + the npz (~1 s).
- **`maxDuration` 60 s**: `/api/best_move` (3 s) and building an analysis line
  move-by-move (~0.6 s/move) are fine; a cold full-game import is ~18 s.
- The API is **stateless** — the Play tab sends `{human_color, history_actions}`
  with every request.
- A separate API host is still possible: set `VITE_API_BASE=https://…` as a
  Vercel build env var and the front end calls that instead of `/api`.

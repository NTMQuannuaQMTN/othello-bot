# Deploying the web app (Vercel)

**One Vercel project.** It builds the Vite front end (static) and serves
`api/index.py` as a Python serverless function on the **same origin**, so the
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
| **Root Directory** | **empty / `./`** (the repo root — *not* `web`, *not* `api`) |
| Framework Preset | Other |
| Build / Output / Install Command | leave "Override" off — they come from `vercel.json` |

Then **Deployments → Redeploy**. Every push to `main` deploys after that.

Use **one** Vercel project — it serves the front end and the `/api/*` function
on the same domain. Delete any separate "api" project.

## Repo layout for the deploy

| path | role |
|---|---|
| `vercel.json` | build command, output dir, function limits, `/api/*` + SPA rewrites |
| `package.json` | `vercel-build` → installs & builds `web/` into `web/dist` |
| `api/index.py` | the function — loads the bot once, delegates to `othello_rl.webapp.server.dispatch` |
| `api/policy.npz` | exported weights (committed, ~1.5 MB) |
| `api/othello_rl/` | the package, **vendored + committed** (~300 KB) so the function bundle is self-contained |
| `api/requirements.txt` | `numpy` |
| `.vercelignore` | keeps torch / checkpoints / data / `pyproject.toml` out of the upload |

## When the model changes

```bash
python3 scripts/export_policy.py     # -> api/policy.npz + re-vendors api/othello_rl
git add api/policy.npz api/othello_rl && git commit -m "web: refresh model"
```

## Run the deploy build locally

```bash
python3 scripts/serve.py --policy api/policy.npz   # torch-free, mirrors Vercel
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

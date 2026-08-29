# OthelloRL

Training an 8×8 Othello / Reversi agent **from scratch** with reinforcement
learning and self-play, and measuring how far it can get against increasingly
strong opponents.

The build order is deliberately incremental:

1. Correct, exhaustively tested Othello engine.
2. Non-RL baseline agents (random, greedy, heuristic, alpha-beta minimax).
3. Tournament + rating evaluation framework.
4. RL environment (canonical observation, action masking, sparse reward).
5. A simple deep-RL agent (masked DQN) — validated to actually learn vs Random.
6. Curriculum training vs fixed opponents.
7. Self-play with a historical opponent pool.
8. Experiment tracking + plots.
9. Terminal play interface.
10. (Later) AlphaZero-style Policy+Value+MCTS upgrade.

See [`PROJECT_SPEC.md`](PROJECT_SPEC.md) for the design, [`TASKS.md`](TASKS.md)
for the checklist, and [`PROGRESS.md`](PROGRESS.md) for current status.

## Setup

Requires Python 3.9+.

```bash
pip install -r requirements.txt
# or: pip install -e ".[dev]"
```

On the reference machine dependencies install into `~/Library/Python/3.9`; make
sure its `bin/` is on `PATH` or invoke tools as `python3 -m <tool>`.

## Running

```bash
export PYTHONPATH=src
python3 -m pytest                              # test suite
python3 scripts/evaluate.py  --config configs/evaluation.yaml
python3 scripts/train.py     --config configs/train.yaml     # live progress bar in a terminal
python3 scripts/selfplay.py  --config configs/selfplay.yaml --init <checkpoint.pt>
python3 scripts/track.py     --run experiments/<run_dir>     # strength curves over checkpoints
python3 scripts/play.py      --checkpoint models/othello_bot_v1.pt   # terminal game
python3 scripts/serve.py     --config configs/webapp.yaml    # web app -> http://127.0.0.1:8000
```

## Web app

`scripts/serve.py` runs a zero-dependency web app (`docs/webapp.md`):

- **Play** against the bot; after a game, **fine-tune** it from your moves —
  the bot's good moves are reinforced and its blunders penalised (graded by a
  1-ply positional check), with a guardrail that rolls back an update that made
  the bot weaker vs a random opponent.
- **Analysis** — Lichess-style move-by-move review: eval graph, per-move
  Best/Inaccuracy/Mistake/Blunder labels, suggested moves.

The bot is a stable component for external testing — `OthelloBot.load(...)`,
or the `scripts/bot_cli.py` line protocol (`genmove` / `eval`). The default
checkpoint is `models/othello_bot_v1.pt` (see `models/README.md`).

`train.py` / `selfplay.py` show a `tqdm` progress bar (total env-steps, current
stage, live epsilon / loss / mean-return) plus periodic eval lines. In a terminal
it's automatic; when output is redirected to a file the bar is replaced by plain
`[stage] N/total steps (pct%)` lines. Force with `--progress on|off`. Watch a
backgrounded run with:

```bash
tail -f experiments/<run_dir>/metrics.jsonl        # structured, one JSON row per eval/chunk
```

## Layout

```
src/othello_rl/
  environment/  board.py rules.py environment.py
  agents/       base.py random_agent.py greedy_agent.py heuristic_agent.py minimax_agent.py
  rl/           network.py replay_buffer.py agent.py trainer.py opponents.py curriculum.py self_play.py
  evaluation/   tournament.py metrics.py elo.py harness.py tracking.py report.py
  utils/        config.py seed.py logging.py experiment.py plots.py progress.py
  webapp/       bot_service.py session.py moves.py server.py static/
scripts/        train.py selfplay.py evaluate.py track.py play.py serve.py bot_cli.py
models/         othello_bot_v1.pt          # the bundled bot
tests/          mirrors src/
```

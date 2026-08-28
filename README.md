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
python3 scripts/play.py      --checkpoint experiments/<run_dir>/checkpoints/final.pt
```

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
  rl/           network.py replay_buffer.py agent.py trainer.py self_play.py opponents.py
  evaluation/   tournament.py metrics.py elo.py
  utils/        config.py seed.py logging.py experiment.py
scripts/        train.py evaluate.py play.py
tests/          mirrors src/
```

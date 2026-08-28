# Project Specification — OthelloRL

## Research question

> How far can an Othello agent trained from scratch through reinforcement learning
> and self-play progress against increasingly strong opponents?

The first delivery target is **not** AlphaZero. It is a correct, reproducible
self-play deep-RL pipeline that demonstrably learns.

## Environment / stack

- Language: Python (developed and tested on CPython 3.9; code targets 3.9+).
  > NOTE: the reference environment for this project only provides Python 3.9.6.
  > The spec suggested 3.11+, but every dependency we need (PyTorch 2.8, NumPy 2.0)
  > installs and runs on 3.9, so we target 3.9+ and avoid 3.10-only syntax.
- Core deps: NumPy, PyTorch, matplotlib, tqdm, PyYAML, pytest.
- `src/` layout, package `othello_rl`.
- Experiment tracking: lightweight JSON/CSV logs + matplotlib plots (no TensorBoard
  dependency required; a `SummaryWriter`-style logger may be added later).

## Board representation

- Board: `numpy.ndarray`, shape `(8, 8)`, dtype `int8`.
- Values: `+1` = Black, `-1` = White, `0` = empty.
- Indexing: `board[row, col]`, `row = 0` is the top rank, `col = 0` is the left file.
- Human notation: files `a`–`h` map to `col` `0`–`7`; ranks `1`–`8` map to `row` `0`–`7`.
  So `"d3"` = `(row=2, col=3)`.
- Standard initial position:
  `(3,3) = White`, `(3,4) = Black`, `(4,3) = Black`, `(4,4) = White`.
- **Black moves first.**
- The eight directions are the 8 king-moves: N, NE, E, SE, S, SW, W, NW.

### Move / action encoding

- A board move is a `(row, col)` tuple.
- The flat action index is `row * 8 + col`, range `0..63`.
- **Pass** is represented as the sentinel action index `64` (and the move value `None`
  at the `Board`/`rules` layer). Pass is legal *only* when the side to move has no
  placing move.

## Rules implemented

Legal-move detection, illegal-move rejection, flipping in all 8 directions,
multi-direction flips from one move, turn switching, forced pass when no legal move,
game termination when neither side can move, final piece-count score, winner
determination (including draws).

## RL design (Phase 4+)

### State (network input)

`float32` tensor, shape `(3, 8, 8)`, always **from the perspective of the side to
move**:

- channel 0: side-to-move's discs (1.0 / 0.0)
- channel 1: opponent's discs
- channel 2: legal-move mask for the side to move (1.0 where a placing move is legal)

This makes the network colour-agnostic; the environment internally canonicalises so
"current player" is always +1 in the observation.

### Action space

`Discrete(65)`: 0..63 = board squares, 64 = pass. Illegal actions are masked with a
boolean mask of length 65 provided in every step/observation. During evaluation the
agent must select only among legal actions.

### Reward

Sparse, zero-sum, from the perspective of the agent whose turn produced the
transition:

- non-terminal step: `0.0`
- win: `+1.0`
- loss: `-1.0`
- draw: `0.0`

No reward shaping in the initial pipeline.

## Algorithm choice (Phase 5)

See `docs/rl-algorithm.md` (written when Phase 5 starts). Planned first algorithm:
**masked Deep Q-Network (DQN)** with a target network and uniform replay — chosen
because the action space is small and discrete, illegal-action masking composes
cleanly with Q-values (mask to `-inf` before argmax / max), and it needs no
opponent-differentiable signal. Policy-gradient / AlphaZero variants come later.

## Reproducibility

Every training/eval run: fixed seed, serialised resolved config, logged git commit,
versioned checkpoints under `experiments/<run-id>/`.

## Definition of done (per task)

Implementation + tests + tests pass + full suite green + matches this spec + docs and
`PROGRESS.md` / `TASKS.md` updated.

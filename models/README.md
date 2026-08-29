# Bundled models

## `othello_bot_v1.pt`

The web app's / test harness's default bot. A masked Double-DQN
(`SmallOthelloNet`, 2 conv blocks × 32 channels, ~410k params) produced by:

1. 380k-step fixed-opponent curriculum (`experiments/20260828-201918_dqn_curriculum`)
2. + 120k-step self-play from a mixed opponent pool
   (`experiments/20260829-012038_dqn_selfplay`)

Strength at export (100 games/opponent, random openings):

| opponent | win rate |
|---|--:|
| Random | 0.93 |
| Greedy | 0.90 |
| Heuristic | 0.36 |

Load it with `DQNAgent.from_checkpoint(...)` or `OthelloBot.load(...)`.
Fine-tuning through the web app writes new versions to `webapp_state/`, never here.

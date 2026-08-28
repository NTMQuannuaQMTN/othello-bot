"""``OthelloEnv`` — a Gymnasium-style environment for the Othello engine.

Design (see ``PROJECT_SPEC.md`` for rationale)
--------------------------------------------
- **Single-agent, alternating control.** Each :meth:`step` plays exactly one ply
  for whichever side is to move. The caller drives *both* sides (self-play). A
  fixed-opponent wrapper lives in :mod:`othello_rl.rl.opponents`.
- **Canonical observation.** ``float32`` array of shape ``(3, 8, 8)`` seen from
  the side to move:

  * channel 0 -- side-to-move discs
  * channel 1 -- opponent discs
  * channel 2 -- legal placing-move mask (1.0 where legal)

- **Action space** ``Discrete(65)``: 0..63 = ``row * 8 + col``; 64 = pass. Pass
  is legal *only* when the side to move has no placing move. ``info["action_mask"]``
  is a length-65 boolean array; illegal actions raise ``ValueError`` unless the
  env was built with ``illegal_move_mode="loss"``.
- **Reward** is sparse and zero-sum, returned from the perspective of the player
  who *just moved*: ``+1`` win, ``-1`` loss, ``0`` draw or non-terminal.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .board import BLACK, BOARD_SIZE, NUM_SQUARES, PASS_ACTION, WHITE, Board
from . import rules

OBS_SHAPE = (3, BOARD_SIZE, BOARD_SIZE)
NUM_ACTIONS = NUM_SQUARES + 1  # 65
MAX_STEPS = 80  # 60 placements + a generous margin for passes


def encode_observation(state: Board) -> np.ndarray:
    """Canonical ``(3, 8, 8)`` float32 observation from ``state.player``'s view."""
    me = (state.array == state.player)
    opp = (state.array == -state.player)
    legal = rules.legal_move_mask(state.array, state.player)
    return np.stack([me, opp, legal]).astype(np.float32)


def legal_action_mask(state: Board) -> np.ndarray:
    """Length-65 boolean mask. Index 64 (pass) is set iff a pass is forced."""
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    moves = state.legal_moves()
    if moves:
        for r, c in moves:
            mask[r * BOARD_SIZE + c] = True
    elif not state.is_terminal():
        mask[PASS_ACTION] = True
    return mask


class OthelloEnv:
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, illegal_move_mode: str = "raise", max_steps: int = MAX_STEPS):
        if illegal_move_mode not in ("raise", "loss"):
            raise ValueError("illegal_move_mode must be 'raise' or 'loss'")
        self.illegal_move_mode = illegal_move_mode
        self.max_steps = max_steps
        self.state: Board = Board.initial()
        self._steps = 0
        self._seed: Optional[int] = None

    # -- gym API --------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self._seed = int(seed)
        self.state = Board.initial()
        self._steps = 0
        return encode_observation(self.state), self._info()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = int(action)
        mover = self.state.player
        mask = legal_action_mask(self.state)

        if not (0 <= action < NUM_ACTIONS) or not mask[action]:
            if self.illegal_move_mode == "loss":
                # end the episode; the mover loses
                self._steps += 1
                info = self._info()
                info["illegal_action"] = action
                return encode_observation(self.state), -1.0, True, False, info
            raise ValueError(
                f"illegal action {action} for {'BLACK' if mover == BLACK else 'WHITE'}; "
                f"legal = {np.nonzero(mask)[0].tolist()}"
            )

        move = None if action == PASS_ACTION else divmod(action, BOARD_SIZE)
        self.state = self.state.apply(move)
        self._steps += 1

        terminated = self.state.is_terminal()
        truncated = (not terminated) and self._steps >= self.max_steps
        reward = 0.0
        if terminated:
            w = self.state.winner()
            reward = 1.0 if w == mover else (-1.0 if w == -mover else 0.0)

        return encode_observation(self.state), reward, terminated, truncated, self._info(mover)

    # -- helpers -------------------------------------------------------
    def _info(self, last_mover: Optional[int] = None) -> Dict[str, Any]:
        mask = legal_action_mask(self.state)
        b, wct = self.state.scores()
        info: Dict[str, Any] = {
            "action_mask": mask,
            "legal_actions": np.nonzero(mask)[0].tolist(),
            "to_play": self.state.player,
            "must_pass": bool(mask[PASS_ACTION]),
            "black_score": b,
            "white_score": wct,
            "steps": self._steps,
        }
        if self.state.is_terminal():
            info["winner"] = self.state.winner()
        if last_mover is not None:
            info["last_mover"] = last_mover
        return info

    def render(self) -> str:
        return self.state.render()

    # -- convenience for tests / rollouts ------------------------------
    @property
    def done(self) -> bool:
        return self.state.is_terminal() or self._steps >= self.max_steps

"""Othello environment: board representation, rules, and RL environment."""
from .board import (
    BLACK,
    BOARD_SIZE,
    EMPTY,
    NUM_SQUARES,
    PASS_ACTION,
    WHITE,
    Board,
    action_to_rc,
    initial_board,
    opponent,
    parse_square,
    rc_to_action,
    render_board,
    square_name,
)
from . import rules
from .environment import (
    NUM_ACTIONS,
    OBS_SHAPE,
    OthelloEnv,
    encode_observation,
    legal_action_mask,
)

__all__ = [
    "OthelloEnv",
    "encode_observation",
    "legal_action_mask",
    "OBS_SHAPE",
    "NUM_ACTIONS",
    "BLACK",
    "WHITE",
    "EMPTY",
    "BOARD_SIZE",
    "NUM_SQUARES",
    "PASS_ACTION",
    "Board",
    "action_to_rc",
    "rc_to_action",
    "initial_board",
    "opponent",
    "parse_square",
    "square_name",
    "render_board",
    "rules",
]

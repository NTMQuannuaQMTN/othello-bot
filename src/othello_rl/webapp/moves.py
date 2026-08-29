"""Parse a game given as a list of moves or an Othello transcript string into a
validated list of action indices from the initial position (passes inserted)."""
from __future__ import annotations

import re
from typing import List, Sequence, Union

from othello_rl.environment.board import Board, PASS_ACTION, parse_square

MoveInput = Union[Sequence, str, dict]


def parse_game(data: MoveInput) -> List[int]:
    if isinstance(data, dict):
        if data.get("history_actions") is not None:
            tokens = list(data["history_actions"])
        elif data.get("moves") is not None:
            tokens = list(data["moves"])
        elif data.get("transcript"):
            tokens = _split_transcript(data["transcript"])
        else:
            tokens = []
    elif isinstance(data, str):
        tokens = _split_transcript(data)
    else:
        tokens = list(data)

    actions: List[int] = []
    state = Board.initial()
    for tok in tokens:
        if state.is_terminal():
            break
        act = _token_to_action(tok)
        legal = state.legal_moves()
        if not legal:
            # forced pass before this move
            actions.append(PASS_ACTION)
            state = state.apply(None)
            if state.is_terminal():
                break
            legal = state.legal_moves()
        if act == PASS_ACTION:
            if legal:
                raise ValueError("pass given but placing moves are available")
            actions.append(PASS_ACTION)
            state = state.apply(None)
            continue
        rc = divmod(act, 8)
        if rc not in legal:
            raise ValueError(f"illegal move {tok!r} at ply {len(actions)}")
        actions.append(act)
        state = state.apply(rc)
    return actions


def replay_positions(actions: Sequence[int]) -> List[dict]:
    """Board state after each ply (index 0 = initial position)."""
    out = []
    state = Board.initial()
    out.append(_pos(state))
    for a in actions:
        a = int(a)
        mv = None if (a == PASS_ACTION or not state.legal_moves()) else divmod(a, 8)
        state = state.apply(mv)
        out.append(_pos(state))
    return out


def _pos(state: Board) -> dict:
    from othello_rl.environment.board import BLACK
    return {
        "grid": [[int(state.array[r, c]) for c in range(8)] for r in range(8)],
        "turn": "black" if state.player == BLACK else "white",
    }


def _token_to_action(tok) -> int:
    if isinstance(tok, (int, float)):
        return int(tok)
    s = str(tok).strip().lower()
    if s in ("pass", "--", "ps"):
        return PASS_ACTION
    r, c = parse_square(s)
    return r * 8 + c


def _split_transcript(text: str) -> List[str]:
    text = text.strip().lower()
    # allow "f5 d6 c3", "f5,d6", or "f5d6c3..."
    parts = re.split(r"[\s,;]+", text)
    if len(parts) > 1:
        return [p for p in parts if p]
    return re.findall(r"[a-h][1-8]|pass", text)

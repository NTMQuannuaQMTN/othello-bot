"""The web app's Othello bot: a thread-safe wrapper around a DQN checkpoint that
can (a) pick moves, (b) analyse a game move-by-move (Lichess-analysis style), and
(c) fine-tune itself from a finished game — rewarding good moves and penalising
blunders, with a guardrail that rolls back an update that made the bot weaker.

The public surface (`OthelloBot`) is deliberately small and stable so an external
harness can test the bot:

    bot = OthelloBot.load("models/othello_bot_v1.pt")
    action = bot.select_action(board)          # 0..63 square, 64 = pass
    move   = bot.select_move(board)            # (row, col) or None
"""
from __future__ import annotations

import copy
import json
import random
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from othello_rl.agents import GreedyAgent, RandomAgent
from othello_rl.agents.heuristic_agent import DEFAULT_WEIGHTS as _HW, evaluate as _heval
from othello_rl.environment.board import (
    BLACK,
    WHITE,
    Board,
    PASS_ACTION,
    action_to_rc,
    opponent,
    square_name,
)
from othello_rl.environment.environment import (
    NUM_ACTIONS,
    encode_observation,
    legal_action_mask,
)
from othello_rl.rl.agent import DQNAgent, NetworkConfig
from othello_rl.rl.replay_buffer import ReplayBuffer

# --------------------------------------------------------------------------- #
# Move-quality classification — chess.com "Expected Points" model
# --------------------------------------------------------------------------- #
#: A move is graded purely by **expected points lost**: EP(best move) − EP(played
#: move), where EP is the mover's win probability after the move (1 = winning,
#: 0.5 = even, 0 = losing). Cutoffs from chess.com's published table; "Best" is
#: reserved for losing (essentially) nothing, i.e. you played the top move.
_CLASS_TABLE: List[Tuple[float, str, str]] = [
    (1e-6, "Best", ""),        # 0.00 expected points lost
    (0.02, "Excellent", ""),   # (0.00, 0.02]
    (0.05, "Good", ""),        # (0.02, 0.05]
    (0.10, "Inaccuracy", "?!"),  # (0.05, 0.10]
    (0.20, "Mistake", "?"),    # (0.10, 0.20]
    (1.001, "Blunder", "??"),  # (0.20, 1.00]
]


def classify_drop(expected_points_lost: float) -> Tuple[str, str]:
    for threshold, label, glyph in _CLASS_TABLE:
        if expected_points_lost < threshold:
            return label, glyph
    return "Blunder", "??"


#: small nudge from the bot's own (noisy) value on top of the positional EP
_BOT_EP_WEIGHT = 0.2
_COACH_SCALE = 18.0  # heuristic-value units -> tanh -> [0, 1)

#: Corners dominate Othello and the small DQN is nearly blind to them, so corner
#: safety is assessed directly and folded into a move's expected points — an
#: X-square move genuinely shows fewer expected points, so it grades badly *and*
#: is never the suggested best move.
_CORNER_RC = ((0, 0), (0, 7), (7, 0), (7, 7))
_CORNER_ACTIONS = frozenset(r * 8 + c for (r, c) in _CORNER_RC)
#: the C- and X-squares guarding each corner (orthogonal neighbours + the diagonal)
_CORNER_ADJ = {
    (0, 0): {(0, 1), (1, 0), (1, 1)},
    (0, 7): {(0, 6), (1, 7), (1, 6)},
    (7, 0): {(7, 1), (6, 0), (6, 1)},
    (7, 7): {(7, 6), (6, 7), (6, 6)},
}
_RISK_OPP_TAKES = 1.0      # opponent can play straight into a corner after this move
_RISK_X_SQUARE = 0.70      # this move sits on the X-square by a still-empty corner
_RISK_C_SQUARE = 0.42      # ... the C-square


def _corner_ep_penalty(risk: float) -> float:
    """Expected points a move gives up purely from corner danger."""
    if risk < 0.0:               # takes a corner
        return -0.06
    if risk >= _RISK_OPP_TAKES:  # opponent can grab a corner next
        return 0.32
    if risk >= _RISK_X_SQUARE:   # X-square next to an empty corner
        return 0.24
    if risk >= _RISK_C_SQUARE:   # C-square
        return 0.11
    return 0.0


def _winprob(value: float) -> float:
    """Map an action/return value in ~[-1, 1] to a win probability in [0, 1]."""
    return float(np.clip((value + 1.0) / 2.0, 0.0, 1.0))


# --------------------------------------------------------------------------- #
@dataclass
class MoveAnalysis:
    ply: int
    side: str                      # "black" / "white"
    played: int                    # action index
    played_san: str
    played_value: float            # bot Q of the played move
    played_winprob: float          # win prob for the side to move after choosing it
    best: int                      # the bot's top move
    best_san: str
    best_value: float
    best_winprob: float
    coach_best_san: str            # a shallow positional check's pick (2nd opinion)
    bot_drop: float                # win-prob lost vs the bot's own best (0..1)
    coach_drop: float              # positional regret, squashed to 0..1
    drop: float                    # combined regret -> feeds the label (0..1)
    label: str
    glyph: str
    eval_after_black: float        # win prob for BLACK after this move (for the graph)
    top_moves: List[dict] = field(default_factory=list)


@dataclass
class FineTuneReport:
    version: int
    games_finetuned: int
    grad_steps: int
    loss_before: float
    loss_after: float
    n_reinforced: int
    n_penalised: int
    winrate_vs_random_before: float
    winrate_vs_random_after: float
    rolled_back: bool
    grades: List[dict]


@dataclass
class FineTuneConfig:
    lr: float = 1e-4
    grad_steps: int = 120
    batch_size: int = 64
    emphasis: int = 6              # times each game transition is added to the buffer
    blunder_penalty: float = 0.8
    great_bonus: float = 0.3
    buffer_capacity: int = 30_000
    anchor_transitions: int = 2_000
    guardrail_games: int = 60
    guardrail_margin: float = 0.10  # roll back if winrate vs random drops by > this


class OthelloBot:
    """Thread-safe bot: move selection, analysis, and self-fine-tuning."""

    def __init__(self, agent: DQNAgent, *, source_path: Optional[str] = None,
                 state_dir: Optional[str] = None, ft_config: Optional[FineTuneConfig] = None,
                 seed: int = 0):
        self.agent = agent
        self.agent.net.eval()
        self.source_path = source_path
        self.ft = ft_config or FineTuneConfig()
        self._lock = threading.RLock()
        self._rng = random.Random(seed)
        # version / lineage survive a restart: they ride in the checkpoint meta,
        # written by `_save_version` and read back here.
        self.version = int(self.agent.meta.extra.get("version", 0))
        self.parent = self.agent.meta.extra.get("parent")
        self.games_finetuned = int(self.agent.meta.extra.get("games_finetuned", 0))
        # the *true* baseline for `reset_to_baseline`: the base checkpoint, not
        # whatever fine-tuned state we happened to load. Only fall back to the
        # loaded weights if the base isn't a distinct readable file.
        self._baseline_is_true = False
        self._baseline_state = self._load_baseline_state(agent)
        self._buffer: Optional[ReplayBuffer] = None
        self.state_dir = Path(state_dir) if state_dir else None
        if self.state_dir:
            (self.state_dir / "history").mkdir(parents=True, exist_ok=True)

    def _load_baseline_state(self, agent: DQNAgent):
        """The weights `reset_to_baseline` restores. Prefer the base checkpoint
        (`source_path`) so a restart after a kept fine-tune still resets to the
        real baseline, not the fine-tuned net."""
        src = self.source_path
        if src and Path(src).is_file():
            try:
                base = DQNAgent.from_checkpoint(src, device=str(agent.device))
                self._baseline_is_true = True
                return copy.deepcopy(base.net.state_dict())
            except Exception:  # pragma: no cover - corrupt/mismatched base
                pass
        self._baseline_is_true = False
        return copy.deepcopy(agent.net.state_dict())

    # -- construction ------------------------------------------------------
    @classmethod
    def load(cls, checkpoint: str, **kw) -> "OthelloBot":
        agent = DQNAgent.from_checkpoint(checkpoint)
        agent.name = "othello-bot"
        kw.setdefault("source_path", str(checkpoint))
        return cls(agent, **kw)

    # -- move selection (the tested interface) ---------------------------
    def select_action(self, board: Board) -> int:
        with self._lock:
            moves = board.legal_moves()
            if not moves:
                return PASS_ACTION
            obs = encode_observation(board)
            mask = legal_action_mask(board)
            return int(self.agent.greedy_act(obs, mask))

    def select_move(self, board: Board):
        a = self.select_action(board)
        return None if a == PASS_ACTION else action_to_rc(a)

    # -- position / move evaluation ------------------------------------
    def _q_values(self, board: Board) -> Tuple[np.ndarray, np.ndarray]:
        obs = encode_observation(board)
        mask = legal_action_mask(board)
        q = self.agent.q_values(obs, mask)  # illegal -> -inf sentinel
        return q, mask

    def _coach_conts(self, board: Board) -> Dict[int, float]:
        """1-ply positional (heuristic) continuation value for every legal move,
        from the mover's perspective — a cheap tactical second opinion."""
        mover = board.player
        out: Dict[int, float] = {}
        for (r, c) in board.legal_moves():
            child = board.apply((r, c))
            if child.player == mover:  # opponent forced to pass
                v = _heval(child.array, mover, _HW)
            else:
                v = -_heval(child.array, child.player, _HW)
            out[r * 8 + c] = float(np.clip(v, -5e4, 5e4))
        return out

    def _corner_risk(self, board: Board, action: int) -> float:
        """How much this move endangers a corner, in [-1, 1]:

        * ``< 0``  the move *takes* a corner;
        * ``0``    corner-neutral;
        * ``0.42`` the move sits on a C-square next to a still-empty corner;
        * ``0.70`` ... the X-square (diagonal) — worse;
        * ``1.0``  after this move the opponent can play straight into a corner.
        """
        if action == PASS_ACTION or not (0 <= action < 64):
            return 0.0
        rc = action_to_rc(action)
        if action in _CORNER_ACTIONS:
            return -1.0
        try:
            child = board.apply(rc)
        except Exception:  # pragma: no cover
            return 0.0
        risk = 0.0
        if not child.is_terminal() and child.player != board.player:
            opp_legal = {r * 8 + c for (r, c) in child.legal_moves()}
            if opp_legal & _CORNER_ACTIONS:
                risk = _RISK_OPP_TAKES
        for corner, adj in _CORNER_ADJ.items():
            if rc in adj and child.array[corner] == 0:
                is_x = abs(rc[0] - corner[0]) == 1 and abs(rc[1] - corner[1]) == 1
                risk = max(risk, _RISK_X_SQUARE if is_x else _RISK_C_SQUARE)
        return risk

    def _corner_flags(self, board: Board, action: int) -> Tuple[bool, bool]:
        """``(takes_corner, gives_corner)`` — a coarse view of ``_corner_risk``."""
        risk = self._corner_risk(board, action)
        return risk < 0.0, risk >= _RISK_X_SQUARE

    def _mover_winprob(self, board: Board) -> float:
        """Positional win probability for the side to move (0..1)."""
        if board.is_terminal():
            w = board.winner()
            if w == 0:
                return 0.5
            return 1.0 if w == board.player else 0.0
        eb = _eval_black(self, board)
        return eb if board.player == BLACK else 1.0 - eb

    def _expected_points(self, board: Board, q: np.ndarray,
                         conts: Dict[int, float]) -> Dict[int, float]:
        """``{action: the mover's expected points (win prob, 0..1) after it}``.

        This one number drives everything — the grade (EP lost = EP(best) −
        EP(played), chess.com's model), the "bot likes" order and the dashed best
        move — so they can never disagree. Corner danger is folded straight in, so
        an X-square move really does show fewer expected points.
        """
        mover = board.player
        out: Dict[int, float] = {}
        for a in conts:
            a = int(a)
            try:
                child = board.apply(action_to_rc(a))
            except Exception:  # pragma: no cover
                continue
            if child.is_terminal():
                w = child.winner()
                base = 1.0 if w == mover else (0.5 if w == 0 else 0.0)
            else:
                m = self._mover_winprob(child)          # positional, child's mover
                base = m if child.player == mover else 1.0 - m
                base = (1.0 - _BOT_EP_WEIGHT) * base + _BOT_EP_WEIGHT * _winprob(float(q[a]))
            base -= _corner_ep_penalty(self._corner_risk(board, a))
            out[a] = float(np.clip(base, 0.0, 1.0))
        return out

    def _ranked_moves(self, board: Board, q: np.ndarray, legal) -> List[int]:
        """Legal moves ordered by expected points (falls back to raw win-prob only
        for a forced-pass position, where there are no continuations)."""
        conts = self._coach_conts(board)
        if not conts:
            return sorted((int(a) for a in legal), key=lambda a: -q[a])
        ep = self._expected_points(board, q, conts)
        return sorted(ep, key=ep.get, reverse=True)

    def grade_move(self, board: Board, played: int) -> dict:
        """Grade one move by **expected points lost** vs the best move
        (chess.com's model): 0 lost -> Best, then Excellent / Good / Inaccuracy /
        Mistake / Blunder per ``_CLASS_TABLE``. Corner danger is already inside
        the expected-points number, so there is no separate override."""
        q, mask = self._q_values(board)
        legal = np.nonzero(mask)[0]
        played = int(played)
        conts = self._coach_conts(board)
        ep = self._expected_points(board, q, conts) if conts else {}

        if ep:
            ranked = sorted(ep, key=ep.get, reverse=True)
            best = ranked[0]
            best_ep = ep[best]
            played_ep = ep.get(played, best_ep)
            ep_lost = max(0.0, best_ep - played_ep)
        else:  # forced pass — nothing to grade against
            ranked = sorted((int(a) for a in legal), key=lambda a: -q[a])
            best = ranked[0] if ranked else PASS_ACTION
            best_ep = played_ep = self._mover_winprob(board)
            ep_lost = 0.0

        coach_best = max(conts, key=conts.get) if conts else best
        coach_drop = (float(np.tanh(max(0.0, conts[coach_best] - conts.get(played, conts[coach_best]))
                                    / _COACH_SCALE)) if conts else 0.0)
        bot_drop = max(0.0, _winprob(float(q[best])) - _winprob(float(q[played]))) \
            if 0 <= played < 64 else 0.0

        crisk = self._corner_risk(board, played)
        takes_c = crisk < 0.0
        gives_c = crisk >= _RISK_X_SQUARE
        # "big loss" for the fine-tuner: this move (not the best) drops into a
        # losing position that best play would have held.
        big_loss = played != best and played_ep < 0.25 and best_ep > 0.4

        label, glyph = classify_drop(ep_lost)
        if label == "Best" and played != best:
            label, glyph = "Excellent", ""

        return {
            "q": q, "mask": mask, "legal": legal, "ranked": [int(a) for a in ranked],
            "bot_best": best, "coach_best": int(coach_best),
            "ep": {int(k): float(v) for k, v in ep.items()},
            "played_ep": float(played_ep), "best_ep": float(best_ep),
            "ep_lost": float(ep_lost),
            "takes_corner": takes_c, "gives_corner": gives_c, "big_loss": big_loss,
            "corner_risk": float(crisk),
            "bot_drop": bot_drop, "coach_drop": coach_drop, "regret": float(ep_lost),
            "label": label, "glyph": glyph,
        }

    def evaluate_position(self, board: Board) -> dict:
        """Bot's read of a position: expected points (win prob) for the side to
        move and for Black, plus the legal moves ranked by expected points."""
        with self._lock:
            if board.is_terminal():
                w = board.winner()
                wp_black = 1.0 if w == BLACK else (0.0 if w == WHITE else 0.5)
                return {"terminal": True, "winner": _side(w) if w else "draw",
                        "winprob_black": wp_black, "winprob_stm": None, "moves": []}
            q, mask = self._q_values(board)
            legal = np.nonzero(mask)[0]
            conts = self._coach_conts(board)
            ep = self._expected_points(board, q, conts) if conts else \
                {int(a): _winprob(float(q[a])) for a in legal}
            ranked = sorted(ep, key=ep.get, reverse=True)
            wp_stm = float(ep[ranked[0]])
            wp_black = _eval_black(self, board)  # positional, for the graph
            moves = []
            for a in ranked:
                risk = self._corner_risk(board, int(a))
                moves.append({
                    "action": int(a),
                    "san": _san(a),
                    "value": float(q[a]),
                    "winprob": float(ep[a]),          # expected points after this move
                    "score": float(ep[a]),
                    "ep_lost": float(max(0.0, ep[ranked[0]] - ep[a])),
                    "corner_risk": float(risk),
                    "gives_corner": risk >= _RISK_X_SQUARE,
                    "takes_corner": risk < 0.0,
                })
            return {"terminal": False, "winprob_black": wp_black,
                    "winprob_stm": wp_stm, "moves": moves}

    def analyse_game(self, actions: Sequence[int], top_k: int = 3) -> List[MoveAnalysis]:
        """Move-by-move analysis of a game given as a list of action indices from
        the initial position."""
        with self._lock:
            out: List[MoveAnalysis] = []
            state = Board.initial()
            for ply, a in enumerate(actions):
                a = int(a)
                if state.is_terminal():
                    break
                if a == PASS_ACTION or not state.legal_moves():
                    state = state.apply(None)
                    continue
                g = self.grade_move(state, a)
                q = g["q"]
                ranked = g["ranked"]
                best = g["bot_best"]
                ep = g["ep"]
                best_wp, played_wp = g["best_ep"], g["played_ep"]
                drop = g["ep_lost"]
                label, glyph = g["label"], g["glyph"]
                nxt = state.apply(a)
                eval_black = _eval_black(self, nxt)
                out.append(MoveAnalysis(
                    ply=ply, side=_side(state.player), played=a, played_san=_san(a),
                    played_value=float(q[a]), played_winprob=played_wp,
                    best=best, best_san=_san(best), best_value=float(q[best]),
                    best_winprob=best_wp,
                    coach_best_san=_san(g["coach_best"]),
                    bot_drop=g["bot_drop"], coach_drop=g["coach_drop"],
                    drop=drop, label=label, glyph=glyph, eval_after_black=eval_black,
                    top_moves=[{"action": int(x), "san": _san(int(x)),
                                "value": float(q[x]),
                                "winprob": float(ep.get(int(x), _winprob(float(q[x]))))}
                               for x in ranked[:top_k]],
                ))
                state = nxt
            return out

    def _position_payload(self, state: Board) -> dict:
        b, w = state.scores()
        moves = state.legal_moves()
        if moves:
            legal = [r * 8 + c for r, c in moves]
        elif not state.is_terminal():
            legal = [PASS_ACTION]
        else:
            legal = []
        return {
            "grid": [[int(state.array[r, c]) for c in range(8)] for r in range(8)],
            "turn": _side(state.player),
            "terminal": state.is_terminal(),
            "winner": ((_side(state.winner()) if state.winner() else "draw")
                       if state.is_terminal() else None),
            "legal_actions": legal,
            "score": {"black": b, "white": w},
            "eval": self.evaluate_position(state),
        }

    def analyse_line(self, actions: Sequence[int], top_k: int = 3) -> dict:
        """Analysis of a line for the interactive (Lichess-style) analysis board:
        one position payload per ply boundary (index 0 = start), plus the grade of
        every played move, an eval graph and a per-side summary."""
        _CORNERS = {(0, 0), (0, 7), (7, 0), (7, 7)}
        _XSQ = {(1, 1), (1, 6), (6, 1), (6, 6)}
        with self._lock:
            state = Board.initial()
            positions = [self._position_payload(state)]
            plies: List[MoveAnalysis] = []
            strat = {"black": {"corners": 0, "x_squares": 0, "edges": 0,
                               "mobility": [], "moves": 0},
                     "white": {"corners": 0, "x_squares": 0, "edges": 0,
                               "mobility": [], "moves": 0}}
            for ply, a in enumerate(actions):
                a = int(a)
                if state.is_terminal():
                    break
                if a == PASS_ACTION or not state.legal_moves():
                    state = state.apply(None)
                    positions.append(self._position_payload(state))
                    continue
                g = self.grade_move(state, a)
                q = g["q"]
                ranked = g["ranked"]
                best = g["bot_best"]

                side = _side(state.player)
                rc = divmod(a, 8)
                sd = strat[side]
                sd["moves"] += 1
                sd["mobility"].append(len(g["legal"]))
                if rc in _CORNERS:
                    sd["corners"] += 1
                elif rc in _XSQ and state.array[_nearest_corner(rc)] == 0:
                    sd["x_squares"] += 1
                elif rc[0] in (0, 7) or rc[1] in (0, 7):
                    sd["edges"] += 1

                nxt = state.apply(a)
                pos = self._position_payload(nxt)
                positions.append(pos)
                ep = g["ep"]
                plies.append(MoveAnalysis(
                    ply=ply, side=_side(state.player), played=a, played_san=_san(a),
                    played_value=float(q[a]), played_winprob=g["played_ep"],
                    best=int(best), best_san=_san(int(best)),
                    best_value=float(q[best]), best_winprob=g["best_ep"],
                    coach_best_san=_san(g["coach_best"]),
                    bot_drop=g["bot_drop"], coach_drop=g["coach_drop"],
                    drop=g["ep_lost"], label=g["label"], glyph=g["glyph"],
                    eval_after_black=pos["eval"]["winprob_black"],
                    top_moves=[{"action": int(x), "san": _san(int(x)),
                                "value": float(q[x]),
                                "winprob": float(ep.get(int(x), _winprob(float(q[x]))))}
                               for x in ranked[:top_k]],
                ))
                state = nxt

            summary: Dict[str, Dict[str, int]] = {"black": {}, "white": {}}
            for p in plies:
                summary[p.side][p.label] = summary[p.side].get(p.label, 0) + 1
            graph = [{"ply": i - 1, "eval_black": positions[i]["eval"]["winprob_black"]}
                     for i in range(len(positions))]

            final = positions[-1]
            fb, fw = final["score"]["black"], final["score"]["white"]
            for side in ("black", "white"):
                sd = strat[side]
                mob = sd.pop("mobility")
                sd["avg_mobility"] = round(sum(mob) / len(mob), 1) if mob else 0.0
                sd["accuracy"] = round(
                    sum(1 for p in plies if p.side == side and p.label in ("Best", "Excellent", "Good"))
                    / max(1, sd["moves"]), 2)
            strat["black"]["final_discs"] = fb
            strat["white"]["final_discs"] = fw
            strat["winner"] = final.get("winner")

            return {
                "actions": [int(a) for a in actions[:len(positions) - 1]],
                "n_moves": len(positions) - 1,  # plies incl. passes
                "positions": positions,
                "plies": [asdict(p) for p in plies],
                "eval_graph": graph,
                "summary": summary,
                "strategy": strat,
            }

    # -- fine-tuning -----------------------------------------------------
    def _ensure_buffer(self) -> ReplayBuffer:
        if self._buffer is None:
            self._buffer = ReplayBuffer(self.ft.buffer_capacity, seed=self._rng.randrange(2**31))
            self._fill_anchor(self.ft.anchor_transitions)
        return self._buffer

    def _fill_anchor(self, n: int) -> None:
        """Seed the replay buffer with the bot's own play vs Random/Greedy so a
        single fine-tune game can't overwrite the whole policy."""
        buf = self._buffer
        opponents = [RandomAgent(seed=1), RandomAgent(seed=2), GreedyAgent()]
        added = 0
        g = 0
        while added < n:
            opp = opponents[g % len(opponents)]
            bot_color = BLACK if g % 2 == 0 else WHITE
            g += 1
            for tr in self._rollout(opp, bot_color):
                buf.add(*tr)
                added += 1

    def _rollout(self, opp, bot_color: int):
        """Play one game (greedy bot vs opp) and yield bot-perspective
        transitions (obs, action, reward, next_obs, done, next_mask)."""
        state = Board.initial()
        transitions = []
        while not state.is_terminal():
            if state.player == bot_color:
                obs = encode_observation(state)
                mask = legal_action_mask(state)
                a = int(self.agent.greedy_act(obs, mask))
                move = None if a == PASS_ACTION else action_to_rc(a)
                nxt = state.apply(move)
                # advance through opponent replies
                while not nxt.is_terminal() and nxt.player != bot_color:
                    om = opp.select_move(nxt)
                    nxt = nxt.apply(om)
                done = nxt.is_terminal()
                r = 0.0
                if done:
                    w = nxt.winner()
                    r = 1.0 if w == bot_color else (-1.0 if w == opponent(bot_color) else 0.0)
                transitions.append((obs, a, r, encode_observation(nxt),
                                    done, legal_action_mask(nxt)))
                state = nxt
            else:
                state = state.apply(opp.select_move(state))
        return transitions

    def _build_game_transitions(self, actions: Sequence[int], learn_color: str):
        """Build DQN transitions from a game for the requested side(s).

        ``learn_color`` is ``"black"``, ``"white"`` or ``"both"`` (every move in
        the list, each side from its own perspective). Returns
        ``(transitions, grades, n_reinforced, n_penalised)``.
        """
        if str(learn_color).lower().startswith(("both", "all")):
            tt, gg, nr, npn = [], [], 0, 0
            for side in ("black", "white"):
                t, g, a, b = self._build_game_transitions(actions, side)
                tt += t; gg += g; nr += a; npn += b
            gg.sort(key=lambda x: x["ply"])
            return tt, gg, nr, npn

        cfg = self.ft
        learn_side = BLACK if str(learn_color).lower().startswith("b") else WHITE
        states = [Board.initial()]
        for a in actions:
            a = int(a)
            mv = None if (a == PASS_ACTION or not states[-1].legal_moves()) else action_to_rc(a)
            states.append(states[-1].apply(mv))
        winner = states[-1].winner() if states[-1].is_terminal() else 0

        grades: List[dict] = []
        trans: List[tuple] = []
        n_reinf = n_pen = 0

        for i, a in enumerate(actions):
            a = int(a)
            s = states[i]
            if s.is_terminal() or s.player != learn_side or not s.legal_moves():
                continue
            g = self.grade_move(s, a)
            label, best, coach_a = g["label"], g["bot_best"], g["coach_best"]
            gives_c, takes_c = g["gives_corner"], g["takes_corner"]
            big_loss, coach_drop = g["big_loss"], g["coach_drop"]

            j = i + 1
            while j < len(states) and not states[j].is_terminal() and states[j].player != learn_side:
                j += 1
            nxt = states[j] if j < len(states) else states[-1]
            done = nxt.is_terminal()
            r = 0.0
            if done:
                r = 1.0 if winner == learn_side else (-1.0 if winner == opponent(learn_side) else 0.0)
            obs, next_obs = encode_observation(s), encode_observation(nxt)
            next_mask = legal_action_mask(nxt)
            trans.append((obs, a, r, next_obs, done, next_mask))

            # Shaping is deliberately conservative: the base transition above
            # already carries the game outcome. We only add an *extra* signal for
            # things we can actually stand behind — conceding vs taking a corner,
            # a clear positional blunder, or the genuine top move in a won game.
            penalised = reinforced = False
            if gives_c and coach_a != a:
                # handing over a corner: hard-penalise it, reward the safe move
                trans.append((obs, a, -cfg.blunder_penalty, next_obs, True, next_mask))
                trans.append((obs, a, -cfg.blunder_penalty, next_obs, True, next_mask))
                trans.append((obs, coach_a, cfg.great_bonus, next_obs, True, next_mask))
                penalised, n_pen = True, n_pen + 1
            elif (label in ("Mistake", "Blunder") or big_loss) and coach_a != a \
                    and coach_drop > 0.08:
                # the positional check clearly disagrees -> penalise, show the fix
                trans.append((obs, a, -cfg.blunder_penalty, next_obs, True, next_mask))
                trans.append((obs, coach_a, cfg.great_bonus, next_obs, True, next_mask))
                penalised, n_pen = True, n_pen + 1
            elif takes_c:
                # taking a corner is (almost) always right — reinforce it
                trans.append((obs, a, max(cfg.great_bonus, r if done else 0.0),
                              next_obs, True, next_mask))
                reinforced, n_reinf = True, n_reinf + 1
            elif label == "Best" and a == best and (coach_a == a or coach_drop < 0.03) \
                    and (r > 0 or not done):
                # unambiguously the best move (both the bot and the check agree)
                # and it didn't lose the game -> a small reinforcement
                trans.append((obs, a, max(cfg.great_bonus, r if done else 0.0),
                              next_obs, True, next_mask))
                reinforced, n_reinf = True, n_reinf + 1

            grades.append({
                "ply": i, "side": _side(s.player), "played": a, "played_san": _san(a),
                "best": int(best), "best_san": _san(int(best)),
                "coach": int(coach_a), "coach_san": _san(int(coach_a)),
                "q_played": float(g["q"][a]), "q_best": float(g["q"][best]),
                "drop": g["regret"], "label": label, "glyph": g["glyph"],
                "penalised": penalised, "reinforced": reinforced,
            })
        return trans, grades, n_reinf, n_pen

    def _run_finetune(self, trans, grades, n_reinf, n_pen, progress=None) -> FineTuneReport:
        """Push ``trans`` into the anchored buffer, train, and keep the update
        only if it doesn't weaken the bot vs Random (paired, same seed)."""
        cfg = self.ft
        buf = self._ensure_buffer()
        for _ in range(cfg.emphasis):
            for tr in trans:
                buf.add(*tr)

        wr_before = self._winrate_vs_random(cfg.guardrail_games)
        prev_state = copy.deepcopy(self.agent.net.state_dict())
        loss_before, loss_after = self._train(cfg, progress)
        wr_after = self._winrate_vs_random(cfg.guardrail_games)

        rolled_back = wr_after < wr_before - cfg.guardrail_margin
        if rolled_back:
            self.agent.net.load_state_dict(prev_state)
            self.agent.net.eval()
        else:
            self.version += 1
            self.games_finetuned += 1
            self.agent.meta.extra["games_finetuned"] = self.games_finetuned
            self.agent.meta.extra["version"] = self.version
            self.agent.meta.extra["parent"] = self.parent
            self._save_version()

        return FineTuneReport(
            version=self.version, games_finetuned=self.games_finetuned,
            grad_steps=cfg.grad_steps, loss_before=loss_before, loss_after=loss_after,
            n_reinforced=n_reinf, n_penalised=n_pen,
            winrate_vs_random_before=wr_before, winrate_vs_random_after=wr_after,
            rolled_back=rolled_back, grades=grades,
        )

    def finetune_from_game(self, actions: Sequence[int], learn_color: str,
                           progress=None) -> FineTuneReport:
        """Fine-tune from one game, learning ``learn_color``'s moves."""
        with self._lock:
            trans, grades, n_reinf, n_pen = self._build_game_transitions(actions, learn_color)
            return self._run_finetune(trans, grades, n_reinf, n_pen, progress)

    def finetune_from_games(self, games: Sequence[dict], progress=None,
                            learn_color: Optional[str] = None) -> FineTuneReport:
        """Fine-tune from many recorded games at once (one training pass, one
        guardrail check).

        ``learn_color`` overrides the side to learn for every game
        (``"black"`` / ``"white"`` / ``"both"``); by default each game teaches
        the bot *its own* moves (opposite of the recorded ``human_color``).
        """
        with self._lock:
            all_trans: List[tuple] = []
            all_grades: List[dict] = []
            n_reinf = n_pen = 0
            for gi, game in enumerate(games):
                actions = game.get("moves") or game.get("actions") or []
                if learn_color:
                    lc = learn_color
                elif game.get("learn_color"):
                    lc = game["learn_color"]
                else:
                    hc = str(game.get("human_color", "white")).lower()
                    lc = "white" if hc.startswith("b") else "black"
                t, gr, nr, np_ = self._build_game_transitions(actions, lc)
                all_trans += t
                for row in gr:
                    row["game"] = gi
                all_grades += gr
                n_reinf += nr
                n_pen += np_
                if progress:
                    progress(gi + 1, len(games))
            return self._run_finetune(all_trans, all_grades, n_reinf, n_pen)

    def _train(self, cfg: FineTuneConfig, progress=None) -> Tuple[float, float]:
        import torch.nn.functional as F
        from othello_rl.rl.network import masked_q

        buf = self._buffer
        net = self.agent.net
        target = self.agent.clone_network()
        opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
        dev = self.agent.device

        def batch_loss(train: bool) -> float:
            b = buf.sample(min(cfg.batch_size, len(buf)))
            obs = torch.as_tensor(b.obs, device=dev)
            act = torch.as_tensor(b.actions, device=dev)
            rew = torch.as_tensor(b.rewards, device=dev)
            nobs = torch.as_tensor(b.next_obs, device=dev)
            done = torch.as_tensor(b.dones, device=dev)
            nmask = torch.as_tensor(b.next_masks, device=dev)
            net.train(train)
            q = net(obs).gather(1, act.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                nq_online = masked_q(net(nobs), nmask)
                na = nq_online.argmax(1, keepdim=True)
                nv = masked_q(target(nobs), nmask).gather(1, na).squeeze(1)
                nv = torch.nan_to_num(nv, neginf=0.0)
                tgt = rew + 0.99 * (1.0 - done) * nv
            loss = F.smooth_l1_loss(q, tgt)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                opt.step()
            return float(loss.item())

        net.eval()
        loss_before = np.mean([batch_loss(False) for _ in range(5)])
        for step in range(cfg.grad_steps):
            batch_loss(True)
            if step % 200 == 199:
                target.load_state_dict(net.state_dict())
            if progress:
                progress(step + 1, cfg.grad_steps)
        net.eval()
        loss_after = np.mean([batch_loss(False) for _ in range(5)])
        return float(loss_before), float(loss_after)

    def _winrate_vs_random(self, n: int) -> float:
        from othello_rl.evaluation.tournament import play_match
        m = play_match(self.agent, "random", num_games=n, seed=99, opening_plies=4)
        return m.a_win_rate

    # -- persistence ---------------------------------------------------
    def _save_version(self) -> None:
        """Persist the fine-tuned scratch model to `state_dir` (never to
        `models/` or `checkpoints/production/` — promotion is script-only)."""
        if not self.state_dir:
            return
        extra = {"version": self.version, "parent": self.parent,
                 "base_checkpoint": self.source_path}
        self.agent.save(self.state_dir / "current.pt", **extra)
        self.agent.save(self.state_dir / "history" / f"v{self.version:04d}.pt", **extra)
        (self.state_dir / "info.json").write_text(json.dumps(self.info(), indent=2))

    def reset_to_baseline(self) -> None:
        with self._lock:
            self.agent.net.load_state_dict(self._baseline_state)
            self.agent.net.eval()
            self.version = 0
            self.games_finetuned = 0
            self.parent = None
            self.agent.meta.extra["games_finetuned"] = 0
            self.agent.meta.extra.pop("version", None)
            self.agent.meta.extra.pop("parent", None)
            self._buffer = None
            if self.state_dir:
                self._save_version()

    def info(self) -> dict:
        nc = self.agent.net_config
        n_params = sum(p.numel() for p in self.agent.net.parameters())
        return {
            "name": self.agent.name,
            "version": self.version,
            "parent": self.parent,
            "games_finetuned": self.games_finetuned,
            "source": self.source_path,
            "baseline": "base-checkpoint" if self._baseline_is_true else "loaded-state",
            "params": int(n_params),
            "network": {"channels": nc.channels, "blocks": nc.blocks, "hidden": nc.hidden},
            "train_env_steps": int(self.agent.meta.env_steps),
        }


# --------------------------------------------------------------------------- #
def _side(player: int) -> str:
    return "black" if player == BLACK else "white"


def _nearest_corner(rc: Tuple[int, int]) -> Tuple[int, int]:
    return (0 if rc[0] < 4 else 7, 0 if rc[1] < 4 else 7)


def _san(action: int) -> str:
    if action == PASS_ACTION:
        return "pass"
    return square_name(action_to_rc(action))


def _eval_black(bot: "OthelloBot", board: Board) -> float:
    """Win probability for BLACK, for the eval bar / graph.

    The DQN's value estimate is STM-relative and near-constant (~0.65 for whoever
    is to move), so blending it in just adds a per-ply zig-zag. The eval line is
    therefore the fast positional score (disc diff, mobility, corners, edges,
    corner danger) from Black's fixed perspective — smooth and directional —
    sharpened as the board fills so a decided endgame reads as ~0/1.
    """
    if board.is_terminal():
        w = board.winner()
        return 1.0 if w == BLACK else (0.0 if w == WHITE else 0.5)
    h = _heval(board.array, BLACK, _HW)  # signed: + = good for black
    filled = int(np.count_nonzero(board.array))
    scale = 26.0 - 14.0 * (filled / 64.0)  # ~26 in the opening -> ~12 late
    return float(np.clip(0.5 + 0.5 * np.tanh(h / scale), 0.02, 0.98))

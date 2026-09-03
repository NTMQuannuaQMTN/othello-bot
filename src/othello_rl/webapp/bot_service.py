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
from othello_rl.rl.numpy_policy import NumpyPolicy
from othello_rl.rl.replay_buffer import ReplayBuffer

#: fine-tuning (and the DQN checkpoint format) needs PyTorch; the deploy build
#: serves with :class:`NumpyPolicy` and never imports it.  ``_require_torch``
#: raises a clean error if a training-only path is hit without it.
try:  # pragma: no cover - trivial
    import torch as _torch  # noqa: F401
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


def _require_torch(what: str):
    if not _HAS_TORCH:
        raise RuntimeError(f"{what} needs the full install (PyTorch); "
                           "this deployment serves inference only.")
    import torch
    return torch


def _load_agent(checkpoint: str):
    """A :class:`NumpyPolicy` for an exported ``.npz`` / model dir (torch-free),
    else a :class:`DQNAgent` from a ``.pt`` checkpoint."""
    p = Path(checkpoint)
    if p.suffix == ".npz" or (p.is_dir() and (p / "policy.npz").is_file()):
        return NumpyPolicy.load(p)
    from othello_rl.rl.agent import DQNAgent
    return DQNAgent.from_checkpoint(checkpoint)

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

#: plies of negamax look-ahead. The eval bar / graph and every move's expected
#: points come from a shallow alpha-beta search (analysis/search.py) so they show
#: who will be better *a few moves from now*, not just the static position.
#: ``_LOOKAHEAD_PLIES`` drives the position eval (one search); ``_EP_LOOKAHEAD``
#: drives each candidate move's expected points (one search per legal move, kept
#: a ply shallower for speed — pure-Python negamax).
_LOOKAHEAD_PLIES = 3
_EP_LOOKAHEAD = 2

#: default per-move search-engine time budget (seconds) for a fresh ``OthelloBot``
#: (move selection, the Play-tab bar). Tests patch this to 0 (engine off / fast).
#: 1.0s + an exact solve from 16 empties (see ``engine_endgame``) is enough to
#: play the endgame perfectly — that is where the policy net used to throw won
#: games with a single move.
_DEFAULT_ENGINE_BUDGET = 1.0
#: per-position budget for whole-line analysis (``analyse_line`` — the suggested
#: move, the eval graph and every grade all come from this one search, for the
#: side to move).  Within one analysis the same position is searched at most once
#: (``_bm_memo``); ``POST /api/best_move`` (3s) is the deepest think.
_ANALYSE_BUDGET = 1.0
#: the exact endgame solve is capped tighter for analysis than for live play —
#: every historical position is analysed and real endgame mistakes start ~13
#: empties.
_ANALYSE_ENDGAME = 13

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
    if risk >= _RISK_X_SQUARE:   # X-square, opponent can force the corner
        return 0.24
    if risk >= _RISK_C_SQUARE:   # C-square, opponent can force the corner
        return 0.11
    if risk > 0.0:               # X-square but the corner is still safe: loose only
        return 0.04
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
    grade_lookahead: int = 3        # plies of look-ahead when judging moves for training


class OthelloBot:
    """Thread-safe bot: move selection, analysis, and self-fine-tuning."""

    def __init__(self, agent, *, source_path: Optional[str] = None,
                 state_dir: Optional[str] = None, ft_config: Optional[FineTuneConfig] = None,
                 seed: int = 0):
        self.agent = agent
        if hasattr(agent, "net"):
            agent.net.eval()
        self.can_finetune = _HAS_TORCH and hasattr(agent, "net")
        self.source_path = source_path
        self.ft = ft_config or FineTuneConfig()
        self._lock = threading.RLock()
        self._rng = random.Random(seed)
        #: search-engine settings for move selection / eval. ``engine_budget``
        #: ``None`` -> :data:`_DEFAULT_ENGINE_BUDGET`; ``<= 0`` turns the engine
        #: off (raw policy — fast tests). ``engine_endgame`` = empties to solve exactly.
        self.engine_budget: Optional[float] = None
        self.engine_endgame = 16
        #: set to a fresh dict while an ``analyse_line`` runs — one whole-line
        #: analysis looks at each position several times (grade, next-position
        #: eval, next ply's grade); this memoises the engine search for that call
        #: and also switches ``best_move`` to the analysis endgame cap.
        self._bm_memo: Optional[dict] = None
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

    def _load_baseline_state(self, agent):
        """The weights `reset_to_baseline` restores. Prefer the base checkpoint
        (`source_path`) so a restart after a kept fine-tune still resets to the
        real baseline, not the fine-tuned net.  ``None`` for a torch-free
        (:class:`NumpyPolicy`) agent — it cannot be fine-tuned anyway."""
        self._baseline_is_true = False
        if not hasattr(agent, "net"):
            return None
        src = self.source_path
        if src and Path(src).is_file() and Path(src).suffix != ".npz":
            try:
                from othello_rl.rl.agent import DQNAgent
                base = DQNAgent.from_checkpoint(src, device=str(agent.device))
                self._baseline_is_true = True
                return copy.deepcopy(base.net.state_dict())
            except Exception:  # pragma: no cover - corrupt/mismatched base
                pass
        return copy.deepcopy(agent.net.state_dict())

    # -- construction ------------------------------------------------------
    @classmethod
    def load(cls, checkpoint: str, **kw) -> "OthelloBot":
        agent = _load_agent(checkpoint)
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

    # -- strong play: a real search engine (alpha-beta + exact endgame) ---
    def best_move(self, board: Board, *, time_budget: Optional[float] = None,
                  endgame_empties: Optional[int] = None) -> dict:
        """The strongest move the bot can find — an alpha-beta search
        (`othello_rl.engine.solver`) that plays the last ``endgame_empties``
        squares out **exactly**.  This is what the web app suggests and what the
        Play-tab bot plays; the DQN policy is only a tiebreak nudge.

        Returns ``{action, san, winprob (Black's), winprob_stm, score, exact,
        depth, nodes, pv}``.  For a forced pass, ``action`` is ``PASS_ACTION``.
        """
        from othello_rl.engine import bitboard as _bb
        from othello_rl.engine.solver import best_move as _search

        budget = time_budget
        if budget is None:
            budget = _DEFAULT_ENGINE_BUDGET if self.engine_budget is None else self.engine_budget
        eg = self.engine_endgame if endgame_empties is None else endgame_empties

        with self._lock:
            if board.is_terminal():
                w = board.winner()
                wb = 1.0 if w == BLACK else (0.0 if w == WHITE else 0.5)
                return {"action": PASS_ACTION, "san": "pass", "winprob": wb,
                        "winprob_stm": None, "score": 0.0, "exact": True,
                        "depth": 0, "nodes": 0, "pv": []}
            if not board.legal_moves():
                return {"action": PASS_ACTION, "san": "pass",
                        "winprob": _eval_black(self, board, {}),
                        "winprob_stm": None, "score": 0.0, "exact": False,
                        "depth": 0, "nodes": 0, "pv": []}
            if budget <= 0:                       # engine off -> raw policy (fast tests)
                a = int(self.agent.greedy_act(encode_observation(board),
                                              legal_action_mask(board)))
                rc = action_to_rc(a)
                stm_wp = _winprob(float(self._q_values(board)[0][a]))
                wb = stm_wp if board.player == BLACK else 1.0 - stm_wp
                return {"action": a, "san": square_name(rc), "winprob": float(wb),
                        "winprob_stm": float(stm_wp), "score": 0.0, "exact": False,
                        "depth": 0, "nodes": 0, "pv": []}
            memo = self._bm_memo
            in_line = memo is not None            # inside analyse_line
            if in_line:
                eg = min(eg, _ANALYSE_ENDGAME)
                budget = _ANALYSE_BUDGET
            mkey = (board.array.tobytes(), int(board.player), int(eg)) if in_line else None
            if mkey is not None and mkey in memo:
                return dict(memo[mkey])
            P, O = _bb.from_grid(board.array, board.player)
            sq, val, meta = _search(P, O, time_budget=budget, endgame_empties=eg)
            rc = action_to_rc(int(sq))
            if meta["exact"]:                                  # margin in discs
                margin = val - (2 ** 20 if val > 0 else -2 ** 20 if val < 0 else 0)
                margin = margin / 1000.0
                stm_wp = 0.5 if abs(margin) < 1e-6 else (0.98 if margin > 0 else 0.02)
            else:
                stm_wp = float(np.clip(0.5 + 0.5 * np.tanh(val / 900.0), 0.02, 0.98))
            wb = stm_wp if board.player == BLACK else 1.0 - stm_wp
            out = {
                "action": int(sq), "san": square_name(rc),
                "winprob": float(wb), "winprob_stm": float(stm_wp),
                "score": float(val if not meta["exact"] else margin),
                "exact": bool(meta["exact"]), "depth": int(meta["depth"]),
                "nodes": int(meta["nodes"]),
                "pv": [square_name(action_to_rc(s)) for s in meta["pv"]],
            }
            if mkey is not None:
                memo[mkey] = dict(out)
            return out

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

    def _corner_forcible(self, board: Board, corner: Tuple[int, int],
                         exchanges: int = 1) -> bool:
        """True when the side to move on ``board`` can *force* taking ``corner``
        within ``exchanges`` more (their-move, opponent-reply) pairs — i.e. even
        with the opponent trying to stop them."""
        if board.array[corner] != 0:
            return False
        ca = corner[0] * 8 + corner[1]
        legal = board.legal_moves()
        if any(r * 8 + c == ca for (r, c) in legal):
            return True
        if exchanges <= 0:
            return False
        taker = board.player
        for tm in legal:                       # taker probes a move
            nb = board.apply(tm)
            if nb.array[corner] != 0:
                continue
            if nb.player == taker:              # opponent had to pass
                if self._corner_forcible(nb, corner, exchanges):
                    return True
                continue
            # opponent to move: the corner is forced iff every reply still loses it
            replies = nb.legal_moves()
            if replies and all(
                    nb.apply(dm).array[corner] == 0
                    and self._corner_forcible(nb.apply(dm), corner, exchanges - 1)
                    for dm in replies):
                return True
        return False

    def _corner_risk(self, board: Board, action: int) -> float:
        """How much this move endangers a corner, in [-1, 1]:

        * ``< 0``   the move *takes* a corner;
        * ``0``     corner-neutral (incl. an X/C-square move where the corner
                    still can't be forced by the opponent);
        * ``0.42``  a C-square move that lets the opponent force the corner;
        * ``0.70``  ... an X-square (diagonal) move — worse;
        * ``1.0``   the opponent can play straight into a corner right now.

        An X/C-square move is only penalised when the opponent can *actually*
        reach the corner — not merely because the square is an X-square.
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
        if child.is_terminal():
            return 0.0
        if child.player != board.player:
            opp_legal = {r * 8 + c for (r, c) in child.legal_moves()}
            if opp_legal & _CORNER_ACTIONS:
                return _RISK_OPP_TAKES

        risk = 0.0
        for corner, adj in _CORNER_ADJ.items():
            if rc not in adj or child.array[corner] != 0:
                continue
            is_x = abs(rc[0] - corner[0]) == 1 and abs(rc[1] - corner[1]) == 1
            if child.player != board.player and self._corner_forcible(child, corner, 1):
                risk = max(risk, _RISK_X_SQUARE if is_x else _RISK_C_SQUARE)
            else:
                risk = max(risk, 0.12 if is_x else 0.0)   # loose, but not losing it
        return risk

    def _corner_flags(self, board: Board, action: int) -> Tuple[bool, bool]:
        """``(takes_corner, gives_corner)`` — a coarse view of ``_corner_risk``."""
        risk = self._corner_risk(board, action)
        return risk < 0.0, risk >= _RISK_X_SQUARE

    def _mover_winprob(self, board: Board, tt: Optional[dict] = None) -> float:
        """Look-ahead win probability for the side to move (0..1)."""
        if board.is_terminal():
            w = board.winner()
            if w == 0:
                return 0.5
            return 1.0 if w == board.player else 0.0
        eb = _eval_black(self, board, tt)
        return eb if board.player == BLACK else 1.0 - eb

    def _expected_points(self, board: Board, q: np.ndarray,
                         conts: Dict[int, float], tt: Optional[dict] = None,
                         lookahead: int = _EP_LOOKAHEAD) -> Dict[int, float]:
        """``{action: the mover's expected points (win prob, 0..1) after it}``,
        from a shallow ``lookahead``-deep negamax search — i.e. how good the move
        looks *a few moves from now*, not just the static position.

        This one number drives everything — the grade (EP lost = EP(best) −
        EP(played), chess.com's model), the "bot likes" order and the dashed best
        move — so they can never disagree. Corner danger is folded straight in.
        """
        from othello_rl.analysis.search import move_value
        scale = _eval_scale(board)
        out: Dict[int, float] = {}
        for a in conts:
            a = int(a)
            try:
                v = move_value(board, action_to_rc(a), int(lookahead) + 1,
                               _HW, tt=tt)             # mover's perspective, heuristic units
            except Exception:  # pragma: no cover
                continue
            base = float(np.clip(0.5 + 0.5 * np.tanh(v / scale), 0.02, 0.98))
            base = (1.0 - _BOT_EP_WEIGHT) * base + _BOT_EP_WEIGHT * _winprob(float(q[a]))
            base -= _corner_ep_penalty(self._corner_risk(board, a))
            out[a] = float(np.clip(base, 0.0, 1.0))
        return out

    def _ranked_moves(self, board: Board, q: np.ndarray, legal,
                      tt: Optional[dict] = None) -> List[int]:
        """Legal moves ordered by expected points (falls back to raw win-prob only
        for a forced-pass position, where there are no continuations)."""
        conts = self._coach_conts(board)
        if not conts:
            return sorted((int(a) for a in legal), key=lambda a: -q[a])
        ep = self._expected_points(board, q, conts, tt)
        return sorted(ep, key=ep.get, reverse=True)

    def _mover_wp_after(self, board: Board, move: int) -> Optional[float]:
        """Win prob for ``board``'s mover *after* they play ``move``, from the
        search engine (exact in the endgame).  ``None`` if the move is illegal."""
        mover = board.player
        legal = {r * 8 + c for r, c in board.legal_moves()}
        if int(move) not in legal:
            return None
        child = board.apply(action_to_rc(int(move)))
        if child.is_terminal():
            w = child.winner()
            return 1.0 if w == mover else 0.0 if w else 0.5
        ce = self.best_move(child)
        stm = float(ce["winprob_stm"])
        # if the opponent was forced to pass, `child`'s mover is still `mover`
        return stm if child.player == mover else 1.0 - stm

    def grade_move(self, board: Board, played: int, tt: Optional[dict] = None,
                   lookahead: int = _EP_LOOKAHEAD) -> dict:
        """Grade one move by **expected points lost** vs the best move
        (chess.com's model): 0 lost -> Best, then Excellent / Good / Inaccuracy /
        Mistake / Blunder per ``_CLASS_TABLE``.

        The best move and both win-probabilities come from the **search engine**
        (:meth:`best_move`, exact endgame) for the side to move — the same source
        as the board's suggested move — so "best" always means best *for whoever
        is on move*, and the move list never disagrees with the board.  The
        shallow ``_expected_points`` is kept only for the secondary "bot likes"
        ordering."""
        q, mask = self._q_values(board)
        legal = np.nonzero(mask)[0]
        played = int(played)
        conts = self._coach_conts(board)
        ep = self._expected_points(board, q, conts, tt, lookahead) if conts else {}

        resolved = (self.engine_budget if self.engine_budget is not None
                    else _DEFAULT_ENGINE_BUDGET)
        eng = self.best_move(board) if (conts and resolved > 0) else None
        if eng and 0 <= int(eng["action"]) < 64:
            best = int(eng["action"])
            best_ep = float(eng["winprob_stm"])
            if played == best:
                played_ep = best_ep
            else:
                pw = self._mover_wp_after(board, played)
                played_ep = best_ep if pw is None else pw
            ep_lost = max(0.0, best_ep - played_ep)
            shallow_rank = sorted(ep, key=ep.get, reverse=True) if ep else []
            ranked = [best] + [a for a in shallow_rank if a != best]
        elif ep:  # engine unavailable (budget 0) — fall back to the shallow score
            ranked = sorted(ep, key=ep.get, reverse=True)
            best = ranked[0]
            best_ep = ep[best]
            played_ep = ep.get(played, best_ep)
            ep_lost = max(0.0, best_ep - played_ep)
        else:  # forced pass — nothing to grade against
            ranked = sorted((int(a) for a in legal), key=lambda a: -q[a])
            best = ranked[0] if ranked else PASS_ACTION
            best_ep = played_ep = self._mover_winprob(board, tt)
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

    def bar_eval(self, board: Board, search_budget: Optional[float] = None) -> dict:
        """Just the eval bar: Black's win probability (from the search engine) +
        the score.  Lighter than :meth:`evaluate_position` (no per-move payload) —
        used by the Play tab's bar."""
        with self._lock:
            b, w = board.scores()
            if board.is_terminal():
                wn = board.winner()
                return {"terminal": True,
                        "winner": _side(wn) if wn else "draw",
                        "winprob_black": 1.0 if wn == BLACK else 0.0 if wn == WHITE else 0.5,
                        "winprob_stm": None, "score": {"black": b, "white": w}, "moves": []}
            eng = self.best_move(board, time_budget=search_budget)
            eb = float(eng["winprob"])
            return {"terminal": False, "winner": None,
                    "winprob_black": eb,
                    "winprob_stm": eb if board.player == BLACK else 1.0 - eb,
                    "score": {"black": b, "white": w}, "moves": [],
                    "engine": {"depth": eng["depth"], "exact": eng["exact"],
                               "best_san": eng["san"], "pv": eng["pv"]}}

    def evaluate_position(self, board: Board, tt: Optional[dict] = None,
                          search_budget: Optional[float] = None) -> dict:
        """Bot's read of a position: expected points (win prob) for the side to
        move and for Black, plus the legal moves ranked by expected points. All
        from a shallow look-ahead — the eval bar shows who leads a few moves on."""
        with self._lock:
            if tt is None:
                tt = {}
            if board.is_terminal():
                w = board.winner()
                wp_black = 1.0 if w == BLACK else (0.0 if w == WHITE else 0.5)
                return {"terminal": True, "winner": _side(w) if w else "draw",
                        "winprob_black": wp_black, "winprob_stm": None, "moves": []}
            q, mask = self._q_values(board)
            legal = np.nonzero(mask)[0]
            conts = self._coach_conts(board)
            ep = self._expected_points(board, q, conts, tt) if conts else \
                {int(a): _winprob(float(q[a])) for a in legal}
            ranked = sorted(ep, key=ep.get, reverse=True)

            # the real engine picks the best move; the shallow `ep` still scores
            # every move for the "bot likes" list + per-move analysis.
            eng = self.best_move(board, time_budget=search_budget) if conts else None
            if eng and eng["action"] in ep:
                best_a = int(eng["action"])
                ranked = [best_a] + [a for a in ranked if a != best_a]
                wp_stm = float(eng["winprob_stm"])
                wp_black = float(eng["winprob"])
            else:
                best_a = ranked[0]
                wp_stm = float(ep[best_a])
                wp_black = _eval_black(self, board, tt)

            ep_best = ep.get(best_a, ep[ranked[0]])
            moves = []
            for a in ranked:
                risk = self._corner_risk(board, int(a))
                # the engine's pick (always ranked[0] when the engine ran) reports
                # the engine's win prob for the side to move, so the board's
                # "best -> X% win" matches the move list's grade; the shallow
                # score of any other move is capped there so the list stays
                # ordered best-first.
                wp = wp_stm if (eng and int(a) == best_a) else (
                    min(float(ep[a]), wp_stm) if eng else float(ep[a]))
                moves.append({
                    "action": int(a),
                    "san": _san(a),
                    "value": float(q[a]),
                    "winprob": float(wp),             # expected points after this move
                    "score": float(wp),
                    "ep_lost": float(max(0.0, ep_best - ep[a])),
                    "corner_risk": float(risk),
                    "gives_corner": risk >= _RISK_X_SQUARE,
                    "takes_corner": risk < 0.0,
                })
            out = {"terminal": False, "winprob_black": wp_black,
                   "winprob_stm": wp_stm, "moves": moves}
            if eng:
                out["engine"] = {"depth": eng["depth"], "exact": eng["exact"],
                                 "nodes": eng["nodes"], "pv": eng["pv"]}
            return out

    def analyse_game(self, actions: Sequence[int], top_k: int = 3) -> List[MoveAnalysis]:
        """Move-by-move analysis of a game given as a list of action indices from
        the initial position."""
        with self._lock:
            out: List[MoveAnalysis] = []
            tt: dict = {}
            state = Board.initial()
            for ply, a in enumerate(actions):
                a = int(a)
                if state.is_terminal():
                    break
                if a == PASS_ACTION or not state.legal_moves():
                    state = state.apply(None)
                    continue
                g = self.grade_move(state, a, tt)
                q = g["q"]
                ranked = g["ranked"]
                best = g["bot_best"]
                ep = g["ep"]
                best_wp, played_wp = g["best_ep"], g["played_ep"]
                drop = g["ep_lost"]
                label, glyph = g["label"], g["glyph"]
                nxt = state.apply(a)
                eval_black = _eval_black(self, nxt, tt)
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

    def _position_payload(self, state: Board, tt: Optional[dict] = None,
                          search_budget: Optional[float] = None) -> dict:
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
            "eval": self.evaluate_position(state, tt, search_budget=search_budget),
        }

    def analyse_line(self, actions: Sequence[int], top_k: int = 3) -> dict:
        """Analysis of a line for the interactive (Lichess-style) analysis board:
        one position payload per ply boundary (index 0 = start), plus the grade of
        every played move, an eval graph and a per-side summary.

        The interactive board re-analyses the *whole* line on every move; a small
        prefix cache means adding one move only searches the new position, not all
        of them (otherwise a long line takes seconds per click)."""
        _CORNERS = {(0, 0), (0, 7), (7, 0), (7, 7)}
        _XSQ = {(1, 1), (1, 6), (6, 1), (6, 6)}
        with self._lock:
            outer = self._bm_memo is None
            if outer:
                self._bm_memo = {}                   # memoise engine searches this call
            try:
                return self._analyse_line(actions, top_k, _CORNERS, _XSQ)
            finally:
                if outer:
                    self._bm_memo = None

    def _analyse_line(self, actions, top_k, _CORNERS, _XSQ) -> dict:
        with self._lock:
            acts = tuple(int(a) for a in actions)
            cache = getattr(self, "_line_cache", None)
            if cache is None:
                from collections import OrderedDict
                cache = self._line_cache = OrderedDict()

            tt: dict = {}
            start = 0
            if acts in cache:
                positions, plies, strat = copy.deepcopy(cache[acts])
                cache.move_to_end(acts)
                start = len(acts)
                state = Board.initial()
                for a in acts:
                    state = state.apply(None if (a == PASS_ACTION or not state.legal_moves())
                                        else action_to_rc(a))
            else:
                prefix = max((k for k in cache if len(k) < len(acts) and acts[:len(k)] == k),
                             key=len, default=None)
                if prefix is not None:
                    positions, plies, strat = copy.deepcopy(cache[prefix])
                    start = len(prefix)
                    state = Board.initial()
                    for a in prefix:
                        state = state.apply(None if (a == PASS_ACTION or not state.legal_moves())
                                            else action_to_rc(a))
                else:
                    state = Board.initial()
                    positions = [self._position_payload(state, tt)]
                    plies = []
                    strat = {"black": {"corners": 0, "x_squares": 0, "edges": 0,
                                       "mobility": [], "moves": 0},
                             "white": {"corners": 0, "x_squares": 0, "edges": 0,
                                       "mobility": [], "moves": 0}}

            for ply, a in enumerate(acts):
                if ply < start:
                    continue
                a = int(a)
                if state.is_terminal():
                    break
                if a == PASS_ACTION or not state.legal_moves():
                    state = state.apply(None)
                    positions.append(self._position_payload(state, tt))
                    continue
                g = self.grade_move(state, a, tt)
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
                pos = self._position_payload(nxt, tt)
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

            # cache the raw (pre-finalise) analysis keyed by this exact line
            cache[acts] = copy.deepcopy((positions, plies, strat))
            while len(cache) > 8:
                cache.popitem(last=False)

            summary: Dict[str, Dict[str, int]] = {"black": {}, "white": {}}
            for p in plies:
                summary[p.side][p.label] = summary[p.side].get(p.label, 0) + 1
            graph = _smoothed_eval_graph(positions, plies)

            strat = copy.deepcopy(strat)      # finalise on a copy; the cache keeps the raw form
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
        tt: dict = {}
        look = int(getattr(self.ft, "grade_lookahead", _EP_LOOKAHEAD))

        for i, a in enumerate(actions):
            a = int(a)
            s = states[i]
            if s.is_terminal() or s.player != learn_side or not s.legal_moves():
                continue
            g = self.grade_move(s, a, tt, lookahead=look)
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
        if not self.can_finetune:
            raise RuntimeError("fine-tuning needs the full install (PyTorch); "
                               "this deployment serves inference only.")
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
        self._line_cache = None            # weights changed -> stale analysis
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
        torch = _require_torch("fine-tuning")
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
        if not self.can_finetune:
            raise RuntimeError("no fine-tune state to reset on this deployment.")
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
            self._line_cache = None
            if self.state_dir:
                self._save_version()

    def info(self) -> dict:
        nc = self.agent.net_config
        g = nc.get if isinstance(nc, dict) else (lambda k, d=None: getattr(nc, k, d))
        if hasattr(self.agent, "net"):
            n_params = int(sum(p.numel() for p in self.agent.net.parameters()))
        else:
            n_params = int(getattr(self.agent, "param_count", 0))
        return {
            "name": self.agent.name,
            "version": self.version,
            "parent": self.parent,
            "games_finetuned": self.games_finetuned,
            "source": self.source_path,
            "baseline": "base-checkpoint" if self._baseline_is_true else "loaded-state",
            "params": n_params,
            "network": {"channels": g("channels"), "blocks": g("blocks"), "hidden": g("hidden")},
            "train_env_steps": int(self.agent.meta.env_steps),
            "can_finetune": self.can_finetune,
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


#: How far the eval bar may move in one ply, by the quality of the move played.
#: A shallow heuristic search genuinely swings a lot on some plies (a corner
#: changes hands, mobility flips) even when the move was the best available — that
#: swing is horizon noise, not information, because the pre-move eval already
#: assumed best play. So a good move is capped hard and only a real mistake is
#: allowed to move the bar freely. The bar still chases the true value, just at
#: most this much per ply, so a genuine multi-ply shift catches up over a few
#: moves instead of jumping.
_EVAL_SWING_CAP: Dict[str, float] = {
    "Best": 0.05, "Excellent": 0.08, "Good": 0.13,
    "Inaccuracy": 0.22, "Mistake": 0.45, "Blunder": 1.0,
}


def _smoothed_eval_graph(positions: List[dict], plies: List["MoveAnalysis"]) -> List[dict]:
    """Eval-graph points (Black win prob) with a per-ply cap on how far the bar
    may move, scaled by the played move's grade (:data:`_EVAL_SWING_CAP`).
    ``positions[i]`` is the state after ``i`` plies (incl. passes); ``positions[i]``
    corresponds to action ``i-1``.  A pass (no entry in ``plies``) is uncapped."""
    raw = [float(p["eval"]["winprob_black"]) for p in positions]
    label_by_action = {int(p.ply): p.label for p in plies}
    out = [{"ply": -1, "eval_black": raw[0], "eval_black_raw": raw[0]}] if raw else []
    smooth = raw[0] if raw else 0.5
    for i in range(1, len(raw)):
        cap = _EVAL_SWING_CAP.get(label_by_action.get(i - 1), 1.0)
        delta = raw[i] - smooth
        smooth = float(np.clip(smooth + max(-cap, min(cap, delta)), 0.02, 0.98))
        out.append({"ply": i - 1, "eval_black": smooth, "eval_black_raw": raw[i]})
    return out


def _eval_scale(board: Board) -> float:
    """tanh scale for turning a heuristic-eval value into a win probability —
    tighter as the board fills, so a decided endgame reads as ~0 / 1."""
    filled = int(np.count_nonzero(board.array))
    return 26.0 - 14.0 * (filled / 64.0)  # ~26 in the opening -> ~12 late


def _eval_black(bot: "OthelloBot", board: Board, tt: Optional[dict] = None) -> float:
    """Win probability for BLACK, for the eval bar / graph.

    A shallow ``_LOOKAHEAD_PLIES``-deep alpha-beta search (heuristic leaf: disc
    diff, mobility, corners, edges, corner danger) — so the bar shows who will be
    ahead a few moves from now, not just the static position. The DQN value is
    left out (it is STM-relative and near-constant, only a per-ply zig-zag).
    """
    if board.is_terminal():
        w = board.winner()
        return 1.0 if w == BLACK else (0.0 if w == WHITE else 0.5)
    from othello_rl.analysis.search import shallow_value
    v = shallow_value(board, _LOOKAHEAD_PLIES, _HW, tt=tt)  # side-to-move perspective
    stm_wp = float(np.clip(0.5 + 0.5 * np.tanh(v / _eval_scale(board)), 0.02, 0.98))
    return stm_wp if board.player == BLACK else 1.0 - stm_wp

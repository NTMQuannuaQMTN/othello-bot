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
# Move-quality classification (Lichess-style)
# --------------------------------------------------------------------------- #
#: (max value-drop vs the best move, label, glyph). "drop" is in win-probability
#: points (0..1); the bot's action-values are mapped q -> (q+1)/2.
_CLASS_TABLE: List[Tuple[float, str, str]] = [
    (0.030, "Best", ""),
    (0.070, "Excellent", ""),
    (0.130, "Good", ""),
    (0.220, "Inaccuracy", "?!"),
    (0.380, "Mistake", "?"),
    (1.001, "Blunder", "??"),
]


def classify_drop(combined_regret: float) -> Tuple[str, str]:
    for threshold, label, glyph in _CLASS_TABLE:
        if combined_regret < threshold:
            return label, glyph
    return "Blunder", "??"


#: how much the bot's own value vs a shallow positional check each count toward
#: a move's "regret" (the number the classification table reads).
_BOT_WEIGHT = 0.5
_COACH_WEIGHT = 0.5
_COACH_SCALE = 18.0  # heuristic-value units -> tanh -> [0, 1)


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
        self.version = 0
        self.games_finetuned = int(self.agent.meta.extra.get("games_finetuned", 0))
        self._baseline_state = copy.deepcopy(self.agent.net.state_dict())
        self._buffer: Optional[ReplayBuffer] = None
        self.state_dir = Path(state_dir) if state_dir else None
        if self.state_dir:
            (self.state_dir / "history").mkdir(parents=True, exist_ok=True)

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

    def grade_move(self, board: Board, played: int) -> dict:
        """Grade one move: bot win-prob regret + positional regret -> label."""
        q, mask = self._q_values(board)
        legal = np.nonzero(mask)[0]
        q_best = int(max(legal, key=lambda x: q[x]))
        bot_drop = max(0.0, _winprob(float(q[q_best])) - _winprob(float(q[played])))

        conts = self._coach_conts(board)
        if played in conts and conts:
            coach_best = max(conts, key=conts.get)
            coach_raw = conts[coach_best] - conts[played]
            coach_drop = float(np.tanh(max(0.0, coach_raw) / _COACH_SCALE))
        else:
            coach_best, coach_drop = q_best, 0.0

        regret = _BOT_WEIGHT * bot_drop + _COACH_WEIGHT * coach_drop
        label, glyph = classify_drop(regret)
        return {
            "q": q, "mask": mask, "legal": legal,
            "bot_best": q_best, "coach_best": int(coach_best),
            "bot_drop": bot_drop, "coach_drop": coach_drop, "regret": regret,
            "label": label, "glyph": glyph,
        }

    def evaluate_position(self, board: Board) -> dict:
        """Bot's read of a position: win prob for the side to move / for black,
        plus the ranked legal moves."""
        with self._lock:
            if board.is_terminal():
                w = board.winner()
                wp_black = 1.0 if w == BLACK else (0.0 if w == WHITE else 0.5)
                return {"terminal": True, "winner": _side(w) if w else "draw",
                        "winprob_black": wp_black, "winprob_stm": None, "moves": []}
            q, mask = self._q_values(board)
            legal = np.nonzero(mask)[0]
            ranked = sorted(legal, key=lambda a: -q[a])
            v = float(q[ranked[0]])
            wp_stm = _winprob(v)
            wp_black = _eval_black(self, board)  # blended, for display
            moves = [{
                "action": int(a),
                "san": _san(a),
                "value": float(q[a]),
                "winprob": _winprob(float(q[a])),
            } for a in ranked]
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
                ranked = sorted(g["legal"], key=lambda x: -q[x])
                best = g["bot_best"]
                best_v, played_v = float(q[best]), float(q[a])
                best_wp, played_wp = _winprob(best_v), _winprob(played_v)
                drop = g["regret"]
                label, glyph = g["label"], g["glyph"]
                nxt = state.apply(a)
                # eval-for-black after the move = bot's read of the resulting position
                eval_black = _eval_black(self, nxt)
                out.append(MoveAnalysis(
                    ply=ply, side=_side(state.player), played=a, played_san=_san(a),
                    played_value=played_v, played_winprob=played_wp,
                    best=best, best_san=_san(best), best_value=best_v, best_winprob=best_wp,
                    coach_best_san=_san(g["coach_best"]),
                    bot_drop=g["bot_drop"], coach_drop=g["coach_drop"],
                    drop=drop, label=label, glyph=glyph, eval_after_black=eval_black,
                    top_moves=[{"action": int(x), "san": _san(int(x)),
                                "value": float(q[x]), "winprob": _winprob(float(q[x]))}
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
                ranked = sorted(g["legal"], key=lambda x: -q[x])
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
                plies.append(MoveAnalysis(
                    ply=ply, side=_side(state.player), played=a, played_san=_san(a),
                    played_value=float(q[a]), played_winprob=_winprob(float(q[a])),
                    best=int(best), best_san=_san(int(best)),
                    best_value=float(q[best]), best_winprob=_winprob(float(q[best])),
                    coach_best_san=_san(g["coach_best"]),
                    bot_drop=g["bot_drop"], coach_drop=g["coach_drop"],
                    drop=g["regret"], label=g["label"], glyph=g["glyph"],
                    eval_after_black=pos["eval"]["winprob_black"],
                    top_moves=[{"action": int(x), "san": _san(int(x)),
                                "value": float(q[x]), "winprob": _winprob(float(q[x]))}
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
        """Grade ``learn_color``'s moves in one game and build DQN transitions
        for them (with blunder / best-move shaping). Returns
        ``(transitions, grades, n_reinforced, n_penalised)``."""
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

            penalised = reinforced = False
            if label in ("Mistake", "Blunder") and coach_a != a:
                trans.append((obs, a, -cfg.blunder_penalty, next_obs, True, next_mask))
                trans.append((obs, coach_a, cfg.great_bonus, next_obs, True, next_mask))
                penalised, n_pen = True, n_pen + 1
            elif label in ("Best", "Excellent") and a == best:
                bonus = max(cfg.great_bonus, r if done else 0.0)
                trans.append((obs, a, bonus, next_obs, True, next_mask))
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

    def finetune_from_games(self, games: Sequence[dict], progress=None) -> FineTuneReport:
        """Fine-tune from many recorded games at once (one training pass, one
        guardrail check). For each game the bot learns *its own* moves
        (``learn_color`` = the opposite of the recorded ``human_color``)."""
        with self._lock:
            all_trans: List[tuple] = []
            all_grades: List[dict] = []
            n_reinf = n_pen = 0
            for gi, game in enumerate(games):
                actions = game.get("moves") or game.get("actions") or []
                if game.get("learn_color"):
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
        if not self.state_dir:
            return
        self.agent.save(self.state_dir / "current.pt")
        self.agent.save(self.state_dir / "history" / f"v{self.version:04d}.pt")
        (self.state_dir / "info.json").write_text(json.dumps(self.info(), indent=2))

    def reset_to_baseline(self) -> None:
        with self._lock:
            self.agent.net.load_state_dict(self._baseline_state)
            self.agent.net.eval()
            self.version = 0
            self.games_finetuned = 0
            self.agent.meta.extra["games_finetuned"] = 0
            self._buffer = None
            if self.state_dir:
                self._save_version()

    def info(self) -> dict:
        nc = self.agent.net_config
        n_params = sum(p.numel() for p in self.agent.net.parameters())
        return {
            "name": self.agent.name,
            "version": self.version,
            "games_finetuned": self.games_finetuned,
            "source": self.source_path,
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
    """Win probability for BLACK, for the eval graph.

    The DQN's value estimates are optimistic and low-variance (V(s) is ~0.65 for
    whoever is to move, regardless of who is actually ahead), which on its own
    produces a meaningless per-ply zig-zag. So the graph blends the bot's estimate
    with the same fast positional score used to grade moves.
    """
    if board.is_terminal():
        w = board.winner()
        return 1.0 if w == BLACK else (0.0 if w == WHITE else 0.5)
    q, _ = bot._q_values(board)
    v = float(np.max(q[np.isfinite(q)]))
    bot_wp = _winprob(v)
    bot_wp_black = bot_wp if board.player == BLACK else 1.0 - bot_wp

    h = _heval(board.array, BLACK, _HW)  # signed: + = good for black
    h_wp_black = float(np.clip(0.5 + 0.5 * np.tanh(h / 22.0), 0.0, 1.0))

    return 0.35 * bot_wp_black + 0.65 * h_wp_black

"""Play the trained RL bot against Egaroucid, with our engine as referee.

Design (see PROJECT_SPEC): our own ``Board`` / rules are the single source of
truth for legality, passing and termination.  Egaroucid is asked only for *its*
moves via GTP ``genmove``; every other move (ours, plus any random opening plies)
is pushed to it with GTP ``play`` so both sides stay in sync.
"""
from __future__ import annotations

import random
import statistics
from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable, List, Optional, Sequence

from othello_rl.environment import rules
from othello_rl.environment.board import (
    BLACK,
    WHITE,
    Board,
    PASS_ACTION,
    action_to_rc,
    opponent,
    rc_to_action,
    square_name,
)

from .egaroucid import EgaroucidEngine, coord_to_gtp

# --------------------------------------------------------------------------- #


def _color_name(player: int) -> str:
    return "black" if player == BLACK else "white"


@dataclass
class GameRecord:
    game_index: int
    rl_color: str                       # "black" / "white"
    egaroucid_color: str
    opening_plies: int
    moves: List[dict] = field(default_factory=list)
    #: real placements only, space-joined SAN (e.g. "e6 f4 c3 ...") — the
    #: portable, replayable transcript. ``moves`` additionally carries ``pass``
    #: annotations, which ``Board.apply`` performs implicitly on replay.
    transcript: str = ""
    winner: str = ""                    # "black" / "white" / "draw"
    result: str = ""                    # "rl_win" / "egaroucid_win" / "draw"
    score_black: int = 0
    score_white: int = 0
    rl_score: int = 0
    egaroucid_score: int = 0
    disc_diff: int = 0                  # rl discs - egaroucid discs
    n_moves: int = 0
    n_passes: int = 0
    rl_inference_times: List[float] = field(default_factory=list)
    egaroucid_result: str = ""          # Egaroucid's own Black-normalised verdict
    egaroucid_agrees: Optional[bool] = None
    error: Optional[str] = None

    def summary_line(self) -> str:
        verdict = {"rl_win": "RL WIN", "egaroucid_win": "RL loss", "draw": "draw"}[self.result]
        return (f"game {self.game_index}: RL={self.rl_color:<5} "
                f"{verdict:<7} {self.score_black}-{self.score_white} "
                f"(RL {self.rl_score}, diff {self.disc_diff:+d}), "
                f"{self.n_moves} plies, {self.n_passes} passes")


class IllegalRLMove(RuntimeError):
    pass


class EngineDesync(RuntimeError):
    pass


def _diagnose_illegal(board: Board, bot, action: int) -> str:
    legal = board.legal_moves()
    lines = [
        "",
        "!!! RL bot proposed an illegal move — stopping the game.",
        board.render(),
        f"side to move : {_color_name(board.player)}",
        f"legal moves  : {[square_name(m) for m in legal] or ['pass']}",
        f"selected     : action {action} -> "
        f"{'pass' if action == PASS_ACTION else square_name(action_to_rc(action))}",
    ]
    q_fn = getattr(getattr(bot, "agent", None), "q_values", None)
    if callable(q_fn):
        try:
            from othello_rl.environment.environment import encode_observation, legal_action_mask
            q = q_fn(encode_observation(board), legal_action_mask(board))
            order = sorted(range(len(q)), key=lambda a: -q[a])[:8]
            lines.append("model output (top Q):")
            for a in order:
                nm = "pass" if a == PASS_ACTION else square_name(action_to_rc(a))
                lines.append(f"    {nm:>4}  q={q[a]:+.4f}")
        except Exception as exc:  # pragma: no cover - diagnostics only
            lines.append(f"(could not read model output: {exc})")
    return "\n".join(lines)


def _cross_check(rec: "GameRecord", printer) -> None:
    """Sanity-check our engine's verdict against the string Egaroucid returned
    from ``gogui-rules_final_result`` (Black-normalised).  A mismatch is a
    warning, not a failure — our engine stays authoritative."""
    text = (rec.egaroucid_result or "").lower()
    if "wins" not in text and "draw" not in text:
        return
    eg_winner = "black" if "black wins" in text else "white" if "white wins" in text else "draw"
    rec.egaroucid_agrees = (eg_winner == rec.winner)
    if not rec.egaroucid_agrees and printer:
        printer(f"  [warn] Egaroucid's final result ({rec.egaroucid_result!r}) "
                f"disagrees with our engine ({rec.winner}); trusting our engine.")


def play_game(bot, engine: EgaroucidEngine, *, rl_is_black: bool,
              game_index: int = 0, opening_plies: int = 0,
              opening_rng: Optional[random.Random] = None,
              verbose: bool = True,
              printer: Callable[[str], None] = print) -> GameRecord:
    """Play one full game.  ``bot`` needs ``select_action(Board) -> int`` (0..63,
    or 64 for a forced pass)."""
    rl_color = BLACK if rl_is_black else WHITE
    rec = GameRecord(
        game_index=game_index,
        rl_color=_color_name(rl_color),
        egaroucid_color=_color_name(-rl_color),
        opening_plies=opening_plies,
    )
    rng = opening_rng or random.Random(game_index)

    engine.clear_board()
    state = Board.initial()
    ply = 0

    def _log(mover: int, actor: str, move, infer_s):
        nonlocal ply
        san = "pass" if move is None else square_name(move)
        rec.moves.append({
            "ply": ply,
            "player": _color_name(mover),
            "actor": actor,
            "san": san,
            "action": PASS_ACTION if move is None else rc_to_action(*move),
            "gtp": coord_to_gtp(move),
            "pass": move is None,
            "inference_s": infer_s,
        })
        if verbose:
            tag = f"  ({infer_s * 1000:.1f} ms)" if infer_s is not None else ""
            printer(f"Move {ply + 1}: {actor} → {san.upper()}{tag}")
        if move is None:
            rec.n_passes += 1
        ply += 1

    def _actor_for(player: int) -> str:
        if ply < opening_plies:
            return "opening"
        return "RL Bot" if player == rl_color else "Egaroucid"

    while not state.is_terminal():
        mover = state.player
        legal = state.legal_moves()          # our engine never lands on an empty-legal
        infer_s = None                        # non-terminal state, but guard anyway

        if not legal:                         # pragma: no cover - defensive
            _log(mover, _actor_for(mover), None, None)
            state = state.apply(None)
            continue

        if ply < opening_plies:
            actor = "opening"
            move = rng.choice(legal)
            engine.play(_color_name(mover), move)
        elif mover == rl_color:
            actor = "RL Bot"
            t0 = perf_counter()
            action = int(bot.select_action(state))
            infer_s = perf_counter() - t0
            rec.rl_inference_times.append(infer_s)
            if action == PASS_ACTION or action_to_rc(action) not in legal:
                msg = _diagnose_illegal(state, bot, action)
                printer(msg)
                rec.error = "illegal RL move"
                raise IllegalRLMove(msg)
            move = action_to_rc(action)
            engine.play(_color_name(mover), move)
        else:
            actor = "Egaroucid"
            eg_move = engine.genmove(_color_name(mover))
            if eg_move is None:
                raise EngineDesync(
                    f"Egaroucid passed at ply {ply} but {_color_name(mover)} has "
                    f"legal moves {[square_name(m) for m in legal]}")
            if eg_move not in legal:
                raise EngineDesync(
                    f"Egaroucid played {coord_to_gtp(eg_move)} at ply {ply}; our "
                    f"engine's legal moves are {[square_name(m) for m in legal]}\n"
                    + state.render())
            move = eg_move

        _log(mover, actor, move, infer_s)
        state = state.apply(move)

        # our Board.apply() auto-skips a blocked opponent; surface that pass
        # explicitly (log it, count it) — Egaroucid handles it internally.
        if not state.is_terminal() and state.player == mover:
            passer = opponent(mover)
            _log(passer, _actor_for(passer), None, None)
        elif state.is_terminal() and not rules.has_any_move(state.array, opponent(mover)):
            b_now, w_now = state.scores()
            if b_now + w_now < 64:            # ended by having no moves, not a full board
                passer = opponent(mover)
                _log(passer, _actor_for(passer), None, None)

    # -- finalise (our engine is authoritative) -------------------------
    b, w = state.scores()
    rec.n_moves = ply
    rec.transcript = " ".join(m["san"] for m in rec.moves if not m["pass"])
    rec.score_black, rec.score_white = int(b), int(w)
    winner = state.winner()
    rec.winner = "draw" if winner == 0 else _color_name(winner)
    rec.rl_score = rec.score_black if rl_is_black else rec.score_white
    rec.egaroucid_score = rec.score_white if rl_is_black else rec.score_black
    rec.disc_diff = rec.rl_score - rec.egaroucid_score
    if winner == 0:
        rec.result = "draw"
    elif winner == rl_color:
        rec.result = "rl_win"
    else:
        rec.result = "egaroucid_win"
    try:
        rec.egaroucid_result = engine.final_result()
    except Exception as exc:  # pragma: no cover
        rec.egaroucid_result = f"(unavailable: {exc})"
    _cross_check(rec, printer if verbose else None)

    if verbose:
        printer(rec.summary_line())
    return rec


@dataclass
class MatchSummary:
    games: int
    rl_wins: int
    egaroucid_wins: int
    draws: int
    win_rate: float                     # wins + 0.5*draws, / games
    rl_mean_score: float
    mean_disc_diff: float
    rl_black_games: int
    rl_white_games: int
    rl_black_wins: int
    rl_white_wins: int
    inference_ms_mean: float
    inference_ms_median: float
    inference_ms_max: float
    rl_total_think_s: float
    total_wall_s: float
    records: List[GameRecord] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["records"] = [r.__dict__ for r in self.records]
        return d


def summarise(records: Sequence[GameRecord], total_wall_s: float) -> MatchSummary:
    all_infer = [t for r in records for t in r.rl_inference_times]
    infer_ms = [t * 1000.0 for t in all_infer]
    rl_wins = sum(r.result == "rl_win" for r in records)
    eg_wins = sum(r.result == "egaroucid_win" for r in records)
    draws = sum(r.result == "draw" for r in records)
    n = len(records)
    return MatchSummary(
        games=n,
        rl_wins=rl_wins,
        egaroucid_wins=eg_wins,
        draws=draws,
        win_rate=(rl_wins + 0.5 * draws) / n if n else 0.0,
        rl_mean_score=statistics.fmean(r.rl_score for r in records) if n else 0.0,
        mean_disc_diff=statistics.fmean(r.disc_diff for r in records) if n else 0.0,
        rl_black_games=sum(r.rl_color == "black" for r in records),
        rl_white_games=sum(r.rl_color == "white" for r in records),
        rl_black_wins=sum(r.rl_color == "black" and r.result == "rl_win" for r in records),
        rl_white_wins=sum(r.rl_color == "white" and r.result == "rl_win" for r in records),
        inference_ms_mean=statistics.fmean(infer_ms) if infer_ms else 0.0,
        inference_ms_median=statistics.median(infer_ms) if infer_ms else 0.0,
        inference_ms_max=max(infer_ms) if infer_ms else 0.0,
        rl_total_think_s=sum(all_infer),
        total_wall_s=total_wall_s,
        records=list(records),
    )


def run_match(bot, engine: EgaroucidEngine, *, games: int = 10,
              opening_plies: int = 4, seed: int = 0,
              start_color: str = "black", verbose: bool = True,
              printer: Callable[[str], None] = print) -> MatchSummary:
    """Play ``games`` games, alternating the RL bot's colour each game."""
    rng = random.Random(seed)
    records: List[GameRecord] = []
    t0 = perf_counter()
    rl_black_first = start_color == "black"
    for i in range(games):
        rl_is_black = (i % 2 == 0) == rl_black_first
        if verbose:
            printer(f"\n=== Game {i + 1}/{games} — RL bot is "
                    f"{'BLACK' if rl_is_black else 'WHITE'} ===")
        opening_rng = random.Random(rng.randrange(2 ** 31))
        rec = play_game(bot, engine, rl_is_black=rl_is_black, game_index=i + 1,
                        opening_plies=opening_plies, opening_rng=opening_rng,
                        verbose=verbose, printer=printer)
        records.append(rec)
    return summarise(records, perf_counter() - t0)


# --------------------------------------------------------------------------- #
# learning from the match (opt-in; scratch model only, never production)
# --------------------------------------------------------------------------- #
def records_to_training_games(records: Sequence[GameRecord]) -> List[dict]:
    """Turn finished :class:`GameRecord`s into the dicts
    ``OthelloBot.finetune_from_games`` expects — real placements only (its
    replay re-inserts forced passes), each game teaching the RL bot *its own*
    side's moves."""
    games: List[dict] = []
    for r in records:
        if r.error:
            continue
        actions = [m["action"] for m in r.moves if not m["pass"]]
        if actions:
            games.append({"actions": actions, "learn_color": r.rl_color})
    return games


def finetune_on_records(bot, records: Sequence[GameRecord], *,
                        grad_steps: Optional[int] = None,
                        lr: Optional[float] = None,
                        guardrail_games: Optional[int] = None):
    """Fine-tune ``bot`` (an ``OthelloBot``) on the games it just played, using
    the project's existing behaviour-cloning + shaping + **guardrail rollback**
    path (`OthelloBot.finetune_from_games`).

    This mutates the in-memory model only. Persisting it is the caller's job and
    must never target `checkpoints/production/` or the registry — the fine-tuned
    net is a *candidate*, evaluated/promoted separately by
    `scripts/{eval_bot,promote_model}.py`.
    """
    if grad_steps is not None:
        bot.ft.grad_steps = int(grad_steps)
    if lr is not None:
        bot.ft.lr = float(lr)
    if guardrail_games is not None:
        bot.ft.guardrail_games = int(guardrail_games)
    games = records_to_training_games(records)
    if not games:
        raise ValueError("no usable games to train on (all had errors)")
    return bot.finetune_from_games(games)

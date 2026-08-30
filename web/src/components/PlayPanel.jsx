import { useCallback, useEffect, useRef, useState } from "react";
import BoardArea from "./BoardArea.jsx";
import FineTuneResult from "./FineTuneResult.jsx";
import { api, cap, sanToIdx } from "../api.js";

const INITIAL_GRID = (() => {
  const g = Array.from({ length: 8 }, () => Array(8).fill(0));
  g[3][3] = -1; g[3][4] = 1; g[4][3] = 1; g[4][4] = -1;
  return g;
})();

export default function PlayPanel({ onBotChanged }) {
  const [game, setGame] = useState(null);   // null until a game is started
  const [apiReady, setApiReady] = useState(false);
  const [evalBlack, setEvalBlack] = useState(0.5);
  const [busy, setBusy] = useState(false);
  const [ft, setFt] = useState({ report: null, error: null, running: false });
  const busyRef = useRef(false);
  const movesEndRef = useRef(null);

  // wait for the API; adopt an in-progress game; honour ?play=black|white|random
  useEffect(() => {
    let stop = false;
    const auto = new URLSearchParams(location.search).get("play");
    const ping = async () => {
      try {
        await api("/bot");
        if (stop) return;
        setApiReady(true);
        if (auto && ["black", "white", "random"].includes(auto)) {
          startGame(auto);
          return;
        }
        const st = await api("/state");           // resume a game across refreshes
        if (!stop && st.ply > 0 && !st.game_over) {
          setGame(st);
          refreshEval();
        }
      } catch {
        if (!stop) setTimeout(ping, 2000);
      }
    };
    ping();
    return () => { stop = true; };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const refreshEval = useCallback(async () => {
    try {
      const e = await api("/eval");
      setEvalBlack(e.winprob_black ?? 0.5);
    } catch { /* ignore */ }
  }, []);

  async function startGame(humanColor) {
    setBusy(true);
    setFt({ report: null, error: null, running: false });
    try {
      const st = await api("/new", { human_color: humanColor });
      setGame(st);
      refreshEval();
    } catch (e) {
      setFt({ report: null, error: e.message, running: false });
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    movesEndRef.current?.scrollIntoView({ block: "nearest" });
  }, [game?.ply]);

  // auto-pass when the human is on move but has no legal placing move
  useEffect(() => {
    if (game && !game.game_over && game.must_pass && game.your_turn && !busyRef.current) {
      move(64);
    }
  }, [game?.ply, game?.must_pass, game?.your_turn, game?.game_over]); // eslint-disable-line react-hooks/exhaustive-deps

  async function move(action) {
    if (busyRef.current) return;
    busyRef.current = true;
    setBusy(true);
    try {
      const st = await api("/move", { action });
      setGame(st);
      await refreshEval();
    } catch (e) {
      setGame((g) => ({ ...g, _err: e.message }));
    } finally {
      busyRef.current = false;
      setBusy(false);
    }
  }

  async function finetune() {
    setFt({ report: null, error: null, running: true });
    try {
      const report = await api("/finetune", {});
      setFt({ report, error: null, running: false });
      onBotChanged?.();
    } catch (e) {
      setFt({ report: null, error: e.message, running: false });
    }
  }

  /* ---------- choose-colour screen ---------- */
  if (!game) {
    return (
      <main>
        <BoardArea grid={INITIAL_GRID} evalBlack={0.5}
          status="Choose your colour to start." />
        <section className="panel">
          <h2 className="start-title">New game</h2>
          <p className="hint">Black moves first.</p>
          <div className="color-choice">
            <button className="primary" disabled={!apiReady || busy}
              onClick={() => startGame("black")}>Play as Black ⚫</button>
            <button disabled={!apiReady || busy}
              onClick={() => startGame("white")}>Play as White ⚪</button>
            <button disabled={!apiReady || busy}
              onClick={() => startGame("random")}>Random</button>
          </div>
          {!apiReady && <p className="hint">waiting for the bot API…</p>}
          {ft.error && <p className="error">{ft.error}</p>}
        </section>
      </main>
    );
  }

  /* ---------- in-game ---------- */
  const lastMoves = (game.last_bot_moves || []).map(sanToIdx).filter((x) => x >= 0);
  let status;
  if (game._err) status = <span className="error">{game._err}</span>;
  else if (game.game_over)
    status = game.winner === "draw"
      ? "Draw."
      : `${cap(game.winner)} wins ${game.score.black}–${game.score.white}.`;
  else status = game.your_turn ? "Your move." : "Bot thinking…";

  const moves = game.moves || [];

  return (
    <main>
      <BoardArea
        grid={game.grid}
        legal={game.your_turn ? game.legal_actions : []}
        last={lastMoves}
        onMove={move}
        evalBlack={evalBlack}
        status={
          <>
            {status} &nbsp;·&nbsp; ⚫ {game.score.black} &nbsp; ⚪ {game.score.white}
            &nbsp;·&nbsp; you are {cap(game.human_color)}
          </>
        }
      />

      <section className="panel">
        <div className="controls">
          <button className="primary" onClick={() => setGame(null)} disabled={busy}>
            New game
          </button>
          <span className="alt">move {game.ply}</span>
        </div>

        <h3 className="mh-title">Move history</h3>
        <ol className="game-moves">
          {moves.length === 0 && <li className="alt">no moves yet</li>}
          {moves.map((m, i) => (
            <li key={i} className={i === moves.length - 1 ? "cur" : ""}>
              <span className="n">{m.n}.</span>
              <span className="disc">{m.side === "black" ? "⚫" : "⚪"}</span>
              <span className="san">{m.pass ? "pass" : m.san}</span>
              <span className="by">{m.by}</span>
            </li>
          ))}
          <li ref={movesEndRef} aria-hidden style={{ display: "block", padding: 0, height: 1 }} />
        </ol>

        {game.game_over && (
          <div className="gameover">
            <p>
              {game.winner === "draw"
                ? "Draw."
                : game.winner === game.human_color
                ? "You won! "
                : "The bot won. "}
              Final {game.score.black}–{game.score.white}.
            </p>
            <button className="primary" onClick={finetune} disabled={ft.running}>
              {ft.running ? "Fine-tuning… (≈5–20s)" : "Fine-tune bot from this game"}
            </button>
            <p className="hint">
              Takes the bot's moves from the game just played, rewards its best moves
              and penalises blunders (graded by a fast positional check), runs a short
              training pass, and keeps the update only if the bot doesn't get weaker
              vs a random opponent.
            </p>
            <FineTuneResult report={ft.report} error={ft.error} />
          </div>
        )}
      </section>
    </main>
  );
}

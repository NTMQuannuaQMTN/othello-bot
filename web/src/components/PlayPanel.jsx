import { useCallback, useEffect, useRef, useState } from "react";
import BoardArea from "./BoardArea.jsx";
import FineTuneResult from "./FineTuneResult.jsx";
import { api, cap, sanToIdx } from "../api.js";

export default function PlayPanel({ onBotChanged }) {
  const [game, setGame] = useState(null);
  const [color, setColor] = useState("black");
  const [evalBlack, setEvalBlack] = useState(0.5);
  const [busy, setBusy] = useState(false);
  const [ft, setFt] = useState({ report: null, error: null, running: false });
  const busyRef = useRef(false);

  const refreshEval = useCallback(async () => {
    try {
      const e = await api("/eval");
      setEvalBlack(e.winprob_black ?? 0.5);
    } catch {
      /* ignore */
    }
  }, []);

  const newGame = useCallback(async () => {
    setFt({ report: null, error: null, running: false });
    const st = await api("/new", { human_color: color });
    setGame(st);
    refreshEval();
  }, [color, refreshEval]);

  // start a game on mount; if the API isn't up yet, keep retrying until it is
  useEffect(() => {
    let stop = false;
    const start = async () => {
      try {
        if (!stop) await newGame();
      } catch {
        if (!stop) setTimeout(start, 2500);
      }
    };
    start();
    return () => { stop = true; };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

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

  if (!game) return <main><p>loading…</p></main>;

  const lastMoves = (game.last_bot_moves || []).map(sanToIdx).filter((x) => x >= 0);
  let status;
  if (game._err) status = <span className="error">{game._err}</span>;
  else if (game.game_over)
    status = game.winner === "draw"
      ? "Draw."
      : `${cap(game.winner)} wins ${game.score.black}–${game.score.white}.`;
  else status = game.your_turn ? "Your move." : "Bot thinking…";

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
          <label>
            You play{" "}
            <select value={color} onChange={(e) => setColor(e.target.value)} disabled={busy}>
              <option value="black">Black (moves first)</option>
              <option value="white">White</option>
              <option value="random">Random</option>
            </select>
          </label>
          <button className="primary" onClick={newGame} disabled={busy}>New game</button>
        </div>

        <ol className="move-list">
          {game.history.map((mv, i) => (
            <li key={i}>{i % 2 === 0 ? "⚫" : "⚪"} <b>{mv}</b></li>
          ))}
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

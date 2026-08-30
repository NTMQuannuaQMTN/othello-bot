import { useCallback, useEffect, useRef, useState } from "react";
import BoardArea from "./BoardArea.jsx";
import MoveList from "./MoveList.jsx";
import FineTuneResult from "./FineTuneResult.jsx";
import { api, cap, sanToIdx } from "../api.js";

const INITIAL_GRID = (() => {
  const g = Array.from({ length: 8 }, () => Array(8).fill(0));
  g[3][3] = -1; g[3][4] = 1; g[4][3] = 1; g[4][4] = -1;
  return g;
})();

export default function PlayPanel({ onBotChanged, onAnalyzeGame }) {
  const [game, setGame] = useState(null);      // null until a game is started
  const [viewPly, setViewPly] = useState(null); // null = live position
  const [apiReady, setApiReady] = useState(false);
  const [evalBlack, setEvalBlack] = useState(0.5);
  const [busy, setBusy] = useState(false);
  const [ft, setFt] = useState({ report: null, error: null, running: false });
  const busyRef = useRef(false);

  const refreshEval = useCallback(async () => {
    try {
      const e = await api("/eval");
      setEvalBlack(e.winprob_black ?? 0.5);
    } catch { /* ignore */ }
  }, []);

  const loadGame = useCallback((st) => {
    setGame(st);
    setViewPly(null);           // snap back to the live position
  }, []);

  async function startGame(humanColor) {
    setBusy(true);
    setFt({ report: null, error: null, running: false });
    try {
      loadGame(await api("/new", { human_color: humanColor }));
      refreshEval();
    } catch (e) {
      setFt({ report: null, error: e.message, running: false });
    } finally {
      setBusy(false);
    }
  }

  // wait for the API; adopt an in-progress game; honour ?play=black|white|random
  useEffect(() => {
    let stop = false;
    const auto = new URLSearchParams(location.search).get("play");
    const ping = async () => {
      try {
        await api("/bot");
        if (stop) return;
        setApiReady(true);
        if (auto && ["black", "white", "random"].includes(auto)) { startGame(auto); return; }
        let st = await api("/state");   // resume/keep the current game across refreshes
        if (stop || st.ply === 0) return;
        // if we refreshed while the bot still owed a reply, collect it
        if (!st.game_over && !st.your_turn) st = await api("/bot_move");
        if (!stop) { loadGame(st); refreshEval(); }
      } catch {
        if (!stop) setTimeout(ping, 2000);
      }
    };
    ping();
    return () => { stop = true; };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // keyboard navigation through history
  useEffect(() => {
    const onKey = (e) => {
      if (!game || e.target.tagName === "INPUT") return;
      const n = game.ply;
      if (e.key === "ArrowLeft") setViewPly((v) => Math.max(0, (v ?? n) - 1));
      else if (e.key === "ArrowRight")
        setViewPly((v) => { const nx = (v ?? n) + 1; return nx >= n ? null : nx; });
      else if (e.key === "Home") setViewPly(0);
      else if (e.key === "End") setViewPly(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [game]);

  // auto-pass when the human is on move but has no legal placing move
  useEffect(() => {
    if (game && viewPly === null && !game.game_over && game.must_pass && game.your_turn && !busyRef.current) {
      move(64);
    }
  }, [game?.ply, game?.must_pass, game?.your_turn, game?.game_over, viewPly]); // eslint-disable-line react-hooks/exhaustive-deps

  const BOT_DELAY_MS = 550;

  async function move(action) {
    if (busyRef.current) return;
    busyRef.current = true;
    setBusy(true);
    try {
      // 1. apply the human move only, so it lands on the board immediately
      let st = await api("/move", { action, bot_reply: false });
      loadGame(st);
      await refreshEval();
      // 2. after a short pause, let the bot reply (may be several plies if we pass)
      if (!st.game_over && !st.your_turn) {
        await new Promise((r) => setTimeout(r, BOT_DELAY_MS));
        st = await api("/bot_move");
        loadGame(st);
        await refreshEval();
      }
    } catch (e) {
      setGame((g) => ({ ...g, _err: e.message }));
    } finally {
      busyRef.current = false;
      setBusy(false);
    }
  }

  const [savedGames, setSavedGames] = useState(0);
  useEffect(() => {
    if (game?.game_over) api("/games").then((d) => setSavedGames(d.count)).catch(() => {});
  }, [game?.game_over, game?.ply]);

  async function finetune(scope) {
    setFt({ report: null, error: null, running: true });
    try {
      const report = await api(scope === "all" ? "/finetune_all" : "/finetune", {});
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
        <BoardArea grid={INITIAL_GRID} evalBlack={0.5} status="Choose your colour to start." />
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
  const moves = game.moves || [];
  const positions = game.positions || [];
  const live = viewPly === null;
  const shownGrid = live ? game.grid : (positions[viewPly] || game.grid);

  // highlight the move that produced the shown position
  const lastMoves = live
    ? (game.last_bot_moves || []).map(sanToIdx).filter((x) => x >= 0)
    : (viewPly > 0 ? [sanToIdx(moves[viewPly - 1]?.san)].filter((x) => x >= 0) : []);

  let status;
  if (!live) status = `Viewing move ${viewPly} of ${game.ply}.`;
  else if (game._err) status = <span className="error">{game._err}</span>;
  else if (game.game_over)
    status = game.winner === "draw"
      ? "Draw." : `${cap(game.winner)} wins ${game.score.black}–${game.score.white}.`;
  else status = game.your_turn ? "Your move." : "Bot thinking…";

  return (
    <main>
      <BoardArea
        grid={shownGrid}
        legal={live && game.your_turn ? game.legal_actions : []}
        last={lastMoves}
        onMove={live ? move : undefined}
        evalBlack={evalBlack}
        status={
          <>
            {status} &nbsp;·&nbsp; ⚫ {game.score.black} &nbsp; ⚪ {game.score.white}
            &nbsp;·&nbsp; you are {cap(game.human_color)}
          </>
        }
        footer={!live && (
          <div className="board-legend">
            <button className="pv" onClick={() => setViewPly(null)}>▶ back to live position</button>
          </div>
        )}
      />

      <section className="panel">
        <div className="controls nav">
          <button className="primary" onClick={() => { setGame(null); setViewPly(null); }} disabled={busy}>
            New game
          </button>
          <span className="spacer" />
          <button onClick={() => setViewPly(0)} disabled={game.ply === 0} title="start">⏮</button>
          <button onClick={() => setViewPly((v) => Math.max(0, (v ?? game.ply) - 1))}
            disabled={game.ply === 0 || viewPly === 0} title="prev (←)">◀</button>
          <button onClick={() => setViewPly((v) => { const n = (v ?? game.ply) + 1; return n >= game.ply ? null : n; })}
            disabled={live} title="next (→)">▶</button>
          <button onClick={() => setViewPly(null)} disabled={live} title="live">⏭</button>
        </div>

        <h3 className="mh-title">Move history <span className="alt">— click a move to view the board</span></h3>
        <MoveList
          items={moves.map((m) => ({
            n: m.n, side: m.side, san: m.pass ? "pass" : m.san, right: m.by,
          }))}
          selected={viewPly ?? game.ply}
          onSelect={(i) => setViewPly(i >= game.ply ? null : i)}
        />

        {game.game_over && (
          <div className="gameover">
            <p>
              {game.winner === "draw" ? "Draw."
                : game.winner === game.human_color ? "You won! " : "The bot won. "}
              Final {game.score.black}–{game.score.white}.
            </p>
            <div className="go-actions">
              <button className="primary" onClick={() => onAnalyzeGame?.(game.history_actions)}>
                Analyse this game
              </button>
              <button onClick={() => finetune()} disabled={ft.running}>
                {ft.running ? "Fine-tuning…" : "Fine-tune from this game"}
              </button>
              {savedGames > 1 && (
                <button onClick={() => finetune("all")} disabled={ft.running}>
                  Fine-tune from all {savedGames} saved games
                </button>
              )}
            </div>
            <p className="hint">
              Every finished game is saved to <code>webapp_state/games.jsonl</code>
              (also usable for offline training — see <code>scripts/finetune_from_games.py</code>).
              <b> Fine-tune</b> rewards the bot's good moves and penalises its blunders,
              runs a short training pass, and keeps the update only if the bot doesn't get
              weaker vs a random opponent.
            </p>
            <FineTuneResult report={ft.report} error={ft.error} />
          </div>
        )}
      </section>
    </main>
  );
}

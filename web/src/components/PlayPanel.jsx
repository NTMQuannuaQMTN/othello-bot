import { useCallback, useEffect, useRef, useState } from "react";
import BoardArea from "./BoardArea.jsx";
import MoveList from "./MoveList.jsx";
import { api, cap, sanToIdx } from "../api.js";

const INITIAL_GRID = (() => {
  const g = Array.from({ length: 8 }, () => Array(8).fill(0));
  g[3][3] = -1; g[3][4] = 1; g[4][3] = 1; g[4][4] = -1;
  return g;
})();

export default function PlayPanel({ onAnalyzeGame }) {
  const [game, setGame] = useState(null);      // null until a game is started
  const [viewPly, setViewPly] = useState(null); // null = live position
  const [apiReady, setApiReady] = useState(false);
  const [evalBlack, setEvalBlack] = useState(0.5);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  const busyRef = useRef(false);

  // the server API is stateless — the client owns the game and sends it every call
  const gameRef = useRef(null);
  const bodyFor = (st, extra) => ({
    human_color: (st || gameRef.current)?.human_color,
    history_actions: (st || gameRef.current)?.history_actions || [],
    ...extra,
  });
  const persist = (st) => {
    try {
      if (st && !st.game_over) {
        localStorage.setItem("othello.game",
          JSON.stringify({ human_color: st.human_color, history_actions: st.history_actions }));
      } else {
        localStorage.removeItem("othello.game");
      }
    } catch { /* private mode */ }
  };

  const refreshEval = useCallback(async (history) => {
    try {
      const e = await api("/eval", { history_actions: history ?? gameRef.current?.history_actions ?? [] });
      setEvalBlack(e.winprob_black ?? 0.5);
    } catch { /* ignore */ }
  }, []);

  const loadGame = useCallback((st) => {
    gameRef.current = st;
    persist(st);
    setGame(st);
    setViewPly(null);           // snap back to the live position
  }, []);

  async function startGame(humanColor) {
    setBusy(true);
    setErr(null);
    try {
      const st = await api("/new", { human_color: humanColor });
      loadGame(st);
      refreshEval(st.history_actions);
    } catch (e) {
      setErr(e.message);
    } finally {
      setBusy(false);
    }
  }

  // wait for the API; resume a game from localStorage; honour ?play=black|white|random
  useEffect(() => {
    let stop = false;
    const auto = new URLSearchParams(location.search).get("play");
    const ping = async () => {
      try {
        await api("/bot");
        if (stop) return;
        setApiReady(true);
        if (auto && ["black", "white", "random"].includes(auto)) { startGame(auto); return; }
        let saved = null;
        try { saved = JSON.parse(localStorage.getItem("othello.game") || "null"); } catch { /* */ }
        if (!saved || !saved.history_actions?.length) return;
        let st = await api("/state", saved);
        if (!stop && !st.game_over && !st.your_turn) st = await api("/bot_move", bodyFor(st));
        if (!stop) { loadGame(st); refreshEval(st.history_actions); }
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
    setErr(null);
    try {
      // 1. apply the human move only, so it lands on the board immediately
      let st = await api("/move", bodyFor(null, { action, bot_reply: false }));
      loadGame(st);
      await refreshEval(st.history_actions);
      // 2. after a short pause, let the bot reply (may be several plies if we pass)
      if (!st.game_over && !st.your_turn) {
        await new Promise((r) => setTimeout(r, BOT_DELAY_MS));
        st = await api("/bot_move", bodyFor(st));
        loadGame(st);
        await refreshEval(st.history_actions);
      }
    } catch (e) {
      setErr(e.message);
    } finally {
      busyRef.current = false;
      setBusy(false);
    }
  }

  // keep the eval bar in sync with the position being viewed (live or history)
  useEffect(() => {
    if (!game) return;
    if (viewPly === null) { refreshEval(); return; }
    const prefix = (game.history_actions || []).slice(0, viewPly);
    api("/eval", { history_actions: prefix })
      .then((e) => setEvalBlack(e.winprob_black ?? 0.5))
      .catch(() => {});
  }, [viewPly, game?.ply, refreshEval]); // eslint-disable-line react-hooks/exhaustive-deps

  function newGame() {
    gameRef.current = null;
    try { localStorage.removeItem("othello.game"); } catch { /* */ }
    setGame(null);
    setViewPly(null);
    setErr(null);
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
          {err && <p className="error">{err}</p>}
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
  else if (err) status = <span className="error">{err}</span>;
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
          <button className="primary" onClick={newGame} disabled={busy}>
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
              <button onClick={newGame}>New game</button>
            </div>
          </div>
        )}
      </section>
    </main>
  );
}

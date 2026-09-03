import { useCallback, useEffect, useMemo, useState } from "react";
import BoardArea from "./BoardArea.jsx";
import EvalGraph from "./EvalGraph.jsx";
import MoveList from "./MoveList.jsx";
import FineTuneResult from "./FineTuneResult.jsx";
import { api, sanToIdx } from "../api.js";

const SUMMARY_ORDER = ["Blunder", "Mistake", "Inaccuracy", "Good", "Excellent", "Best"];
const EMPTY = { positions: [startPosition()], plies: [], eval_graph: [{ ply: -1, eval_black: 0.5 }], summary: { black: {}, white: {} } };

export default function AnalysisPanel({ loadLine, onBotChanged }) {
  const [line, setLine] = useState([]);
  const [cursor, setCursor] = useState(0);
  const [data, setData] = useState(EMPTY);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [showImport, setShowImport] = useState(false);
  const [importText, setImportText] = useState("");
  const [ft, setFt] = useState({ report: null, error: null, running: null });
  const [saved, setSaved] = useState(null);

  async function saveGame() {
    try {
      const r = await api("/games", { moves: line });
      setSaved(r.saved ? `saved · ${r.count} games` : `${r.reason} · ${r.count} games`);
    } catch (e) {
      setSaved(e.message);
    }
  }

  async function learnFrom(color) {
    setFt({ report: null, error: null, running: color });
    try {
      const report = await api("/finetune", { moves: line, learn_color: color });
      setFt({ report, error: null, running: null });
      onBotChanged?.();
    } catch (e) {
      setFt({ report: null, error: e.message, running: null });
    }
  }

  const analyse = useCallback(async (moves) => {
    setBusy(true);
    setError(null);
    try {
      const d = await api("/analyse", { moves });
      setData(d);
      setCursor(d.positions.length - 1); // jump to the latest position
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }, []);

  useEffect(() => { analyse(line); setSaved(null); }, [line, analyse]);

  // deep link: ?analyse=<transcript>  (one-time, on mount)
  useEffect(() => {
    const p = new URLSearchParams(location.search);
    const t = p.get("analyse") || p.get("analysis");
    if (!t) return;
    api("/analyse", { transcript: t })
      .then((d) => setLine(d.actions || []))
      .catch((e) => setError(e.message));
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // "Analyse this game" from the Play tab
  useEffect(() => {
    if (loadLine && Array.isArray(loadLine.actions)) setLine(loadLine.actions);
  }, [loadLine]);

  // keyboard navigation
  useEffect(() => {
    const onKey = (e) => {
      if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
      const n = data.positions.length;
      if (e.key === "ArrowLeft") setCursor((c) => Math.max(0, c - 1));
      else if (e.key === "ArrowRight") setCursor((c) => Math.min(n - 1, c + 1));
      else if (e.key === "Home") setCursor(0);
      else if (e.key === "End") setCursor(n - 1);
      else if (e.key === "Backspace") takeBack();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }); // eslint-disable-line react-hooks/exhaustive-deps

  const pos = data.positions[Math.min(cursor, data.positions.length - 1)];
  const ply = cursor > 0 ? data.plies[cursor - 1] : null;
  const graphPoints = useMemo(() => data.eval_graph.map((g) => g.eval_black), [data]);

  function play(action) {
    // truncate any continuation past the cursor, then append
    setLine((l) => [...l.slice(0, cursor), action]);
  }
  function takeBack() {
    setLine((l) => (l.length ? l.slice(0, -1) : l));
  }
  function clearLine() {
    setLine([]);
  }
  async function useLastGame() {
    try {
      const st = await api("/state");
      setLine(st.history_actions || []);
    } catch (e) {
      setError(e.message);
    }
  }
  async function importTranscript() {
    try {
      setError(null);
      const d = await api("/analyse", { transcript: importText });
      setLine(d.actions || []);
      setShowImport(false);
    } catch (e) {
      setError(e.message);
    }
  }

  // board decorations
  const engineMoves = pos.eval && pos.eval.moves ? pos.eval.moves : [];
  // ALWAYS highlight the bot's top move among all legal moves at this position
  const bestForBoard = pos.terminal || !engineMoves.length ? null : engineMoves[0].action;
  const glyphs = {};
  if (ply) glyphs[ply.played] = { label: ply.label, glyph: ply.glyph };

  const status = statusLine(pos, ply, engineMoves);

  return (
    <main>
      <BoardArea
        grid={pos.grid}
        legal={pos.terminal ? [] : pos.legal_actions}
        last={ply ? [ply.played] : []}
        best={bestForBoard}
        glyphs={glyphs}
        onMove={play}
        evalBlack={graphPoints[cursor]}
        status={status}
        footer={!pos.terminal && engineMoves.length > 0 && (
          <div className="board-legend">
            <span><i className="lg-best" /> best for {sideLabel(pos.turn)}:{" "}
              {engineMoves[0].san} &rarr; {pct(engineMoves[0].winprob)} win</span>
            {ply && ply.played === ply.best && (
              <span><b className="label Best">✓</b> {sideLabel(ply.side)} played the best move</span>
            )}
            {ply && ply.played !== ply.best && (
              <span><i className="lg-last" /> {sideLabel(ply.side)} played {ply.played_san}{" "}
                — best was {ply.best_san}
                <b className={"label " + ply.label}>
                  {" "}{ply.label}{ply.drop > 0.001 ? ` −${Math.round(ply.drop * 100)}` : ""}</b>
              </span>
            )}
          </div>
        )}
      />

      <section className="panel">
        <div className="controls nav">
          <button onClick={() => setCursor(0)} disabled={cursor === 0} title="start">⏮</button>
          <button onClick={() => setCursor((c) => Math.max(0, c - 1))} disabled={cursor === 0} title="prev (←)">◀</button>
          <button onClick={() => setCursor((c) => Math.min(data.positions.length - 1, c + 1))}
            disabled={cursor >= data.positions.length - 1} title="next (→)">▶</button>
          <button onClick={() => setCursor(data.positions.length - 1)}
            disabled={cursor >= data.positions.length - 1} title="end">⏭</button>
          <span className="spacer" />
          <button onClick={takeBack} disabled={!line.length} title="take back (Backspace)">take back</button>
          <button onClick={clearLine} disabled={!line.length}>clear</button>
          <button onClick={useLastGame}>use last game</button>
          <button onClick={() => setShowImport((v) => !v)}>{showImport ? "close" : "paste game"}</button>
          {busy && <span className="alt">analysing…</span>}
        </div>

        {showImport && (
          <div className="controls">
            <input type="text" placeholder="f5 d6 c3 …  or  f5d6c3…"
              value={importText} onChange={(e) => setImportText(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && importTranscript()} />
            <button className="primary" onClick={importTranscript}>Load</button>
          </div>
        )}
        {error && <p className="error">{error}</p>}

        {!pos.terminal && engineMoves.length > 0 && (
          <div className="engine-lines">
            <span className="alt">bot likes:</span>
            {engineMoves.slice(0, 4).map((m) => (
              <button key={m.action} className="pv" onClick={() => play(m.action)}>
                {m.san} <span className="alt">{Math.round(m.winprob * 100)}%</span>
              </button>
            ))}
          </div>
        )}

        <EvalGraph points={graphPoints} cursor={cursor} onSeek={setCursor} />
        <p className="alt eval-caption">
          Win probability for ⚫ Black from a short look-ahead search — who will be
          ahead a few moves from now, not just the current count.
        </p>

        <div className="analysis-summary">
          {["black", "white"].map((side) => {
            const c = data.summary[side] || {};
            const parts = SUMMARY_ORDER.filter((k) => c[k]);
            const s = data.strategy?.[side];
            return (
              <div key={side}>
                <div>
                  {side === "black" ? "⚫ Black" : "⚪ White"}:{" "}
                  {parts.length
                    ? parts.map((k) => (
                        <span key={k}><b className={"label " + k}>{c[k]}</b> {k}&nbsp;&nbsp;</span>
                      ))
                    : "—"}
                </div>
                {s && s.moves > 0 && (
                  <div className="strategy">
                    accuracy {Math.round(s.accuracy * 100)}% · {s.corners} corner
                    {s.corners === 1 ? "" : "s"}
                    {s.x_squares ? ` · ${s.x_squares} X-square${s.x_squares === 1 ? "" : "s"}` : ""}
                    {" "}· {s.edges} edge · avg {s.avg_mobility} moves free · {s.final_discs} discs
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {line.length > 3 && (
          <div className="learn-from">
            <span className="alt">Teach the bot this game:</span>
            <button disabled={!!ft.running} onClick={saveGame}>
              {saved ? saved : "Save to dataset"}
            </button>
            <button className="primary" disabled={!!ft.running} onClick={() => learnFrom("both")}>
              {ft.running === "both" ? "learning…" : "Learn the whole game"}
            </button>
            <button disabled={!!ft.running} onClick={() => learnFrom("black")}>
              {ft.running === "black" ? "learning…" : "⚫ Black only"}
            </button>
            <button disabled={!!ft.running} onClick={() => learnFrom("white")}>
              {ft.running === "white" ? "learning…" : "⚪ White only"}
            </button>
          </div>
        )}
        {(ft.report || ft.error) && (
          <FineTuneResult report={ft.report} error={ft.error} />
        )}

        <MoveList
          items={data.plies.map((p) => {
            // win% for the mover after the move played, and the best alternative
            const w = pct(p.played_winprob);
            let note;
            if (p.best === p.played) note = `${w} win · ✓ best`;
            else note = `${w} win · best ${p.best_san} ${pct(p.best_winprob)}` +
              (p.drop > 0.001 ? ` (−${Math.round(p.drop * 100)})` : "");
            return {
              n: p.ply + 1, side: p.side, san: p.played_san, note,
              right: p.glyph || p.label, rightClass: "label " + p.label,
            };
          })}
          selected={cursor}
          onSelect={setCursor}
          emptyText="Play moves on the board to build a line — the bot analyses each one."
        />
      </section>
    </main>
  );
}

function pct(p) {
  return `${Math.round((p ?? 0.5) * 100)}%`;
}

function sideLabel(side) {
  return side === "white" ? "⚪ White" : "⚫ Black";
}

function statusLine(pos, ply, engineMoves) {
  if (pos.terminal) {
    return pos.winner === "draw"
      ? `Game over — draw ${pos.score.black}–${pos.score.white}.`
      : `Game over — ${pos.winner} wins ${pos.score.black}–${pos.score.white}.`;
  }
  const top = engineMoves.slice(0, 3)
    .map((m) => `${m.san} ${pct(m.winprob)}`).join(", ");
  if (!ply) return `Starting position. Best for ${sideLabel(pos.turn)}: ${top}`;
  return `${sideLabel(ply.side)} played ${ply.played_san} — ${ply.label} ` +
    `(${pct(ply.played_winprob)} win` +
    (ply.best !== ply.played
      ? `, best was ${ply.best_san} ${pct(ply.best_winprob)})`
      : `)`) +
    `.  Best for ${sideLabel(pos.turn)} now: ${top}`;
}

function startPosition() {
  const g = Array.from({ length: 8 }, () => Array(8).fill(0));
  g[3][3] = -1; g[3][4] = 1; g[4][3] = 1; g[4][4] = -1;
  return {
    grid: g, turn: "black", terminal: false, winner: null,
    legal_actions: [19, 26, 37, 44], score: { black: 2, white: 2 },
    eval: { terminal: false, winprob_black: 0.5, winprob_stm: 0.5, moves: [] },
  };
}


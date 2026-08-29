import { useCallback, useEffect, useMemo, useState } from "react";
import BoardArea from "./BoardArea.jsx";
import EvalGraph from "./EvalGraph.jsx";
import { api, sanToIdx } from "../api.js";

const SUMMARY_ORDER = ["Blunder", "Mistake", "Inaccuracy", "Good", "Excellent", "Best"];
const EMPTY = { positions: [startPosition()], plies: [], eval_graph: [{ ply: -1, eval_black: 0.5 }], summary: { black: {}, white: {} } };

export default function AnalysisPanel() {
  const [line, setLine] = useState([]);
  const [cursor, setCursor] = useState(0);
  const [data, setData] = useState(EMPTY);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [showImport, setShowImport] = useState(false);
  const [importText, setImportText] = useState("");

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

  useEffect(() => { analyse(line); }, [line, analyse]);

  // deep link: ?analyse=<transcript>  (one-time, on mount)
  useEffect(() => {
    const p = new URLSearchParams(location.search);
    const t = p.get("analyse") || p.get("analysis");
    if (!t) return;
    api("/analyse", { transcript: t })
      .then((d) => setLine(d.actions || []))
      .catch((e) => setError(e.message));
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

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
  const best = engineMoves.length ? engineMoves[0].action : null;
  const glyphs = {};
  let bestForBoard = pos.terminal ? null : best;
  if (ply) {
    glyphs[ply.played] = { label: ply.label, glyph: ply.glyph };
    if (ply.best !== ply.played) bestForBoard = ply.best;
    else if (ply.label !== "Best" && ply.label !== "Excellent")
      bestForBoard = sanToIdx(ply.coach_best_san);
  }

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

        <div className="analysis-summary">
          {["black", "white"].map((side) => {
            const c = data.summary[side] || {};
            const parts = SUMMARY_ORDER.filter((k) => c[k]);
            return (
              <div key={side}>
                {side === "black" ? "⚫ Black" : "⚪ White"}:{" "}
                {parts.length
                  ? parts.map((k) => (
                      <span key={k}><b className={"label " + k}>{c[k]}</b> {k}&nbsp;&nbsp;</span>
                    ))
                  : "—"}
              </div>
            );
          })}
        </div>

        <ol className="analysis-list">
          {data.plies.map((p) => {
            let alt = "";
            if (p.label !== "Best" && p.label !== "Excellent") {
              const s = p.best_san !== p.played_san ? p.best_san
                : p.coach_best_san !== p.played_san ? p.coach_best_san : null;
              if (s) alt = `try ${s} · −${Math.round(p.drop * 100)}`;
            }
            return (
              <li key={p.ply} className={cursor === p.ply + 1 ? "sel" : ""}
                onClick={() => setCursor(p.ply + 1)}>
                <span className="alt">{p.ply + 1}.</span>
                <span className="mv">{p.side === "black" ? "⚫" : "⚪"}{p.played_san}</span>
                <span className="alt">{alt}</span>
                <span className={"label " + p.label}>{p.glyph || p.label}</span>
              </li>
            );
          })}
          {!data.plies.length && (
            <li className="alt" style={{ display: "block", cursor: "default" }}>
              Play moves on the board to build a line — the bot analyses each one.
            </li>
          )}
        </ol>
      </section>
    </main>
  );
}

function statusLine(pos, ply, engineMoves) {
  if (pos.terminal) {
    return pos.winner === "draw"
      ? `Game over — draw ${pos.score.black}–${pos.score.white}.`
      : `Game over — ${pos.winner} wins ${pos.score.black}–${pos.score.white}.`;
  }
  const top = engineMoves.slice(0, 3)
    .map((m) => `${m.san} ${Math.round(m.winprob * 100)}%`).join(", ");
  if (!ply) return `Starting position. Bot likes: ${top}`;
  return `${ply.side} played ${ply.played_san} — ${ply.label}` +
    (ply.best_san !== ply.played_san ? ` (best: ${ply.best_san})` : "") +
    `.  Now: ${top}`;
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


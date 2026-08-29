import { useEffect, useMemo, useState } from "react";
import BoardArea from "./BoardArea.jsx";
import EvalGraph from "./EvalGraph.jsx";
import { api, sanToIdx } from "../api.js";

const SUMMARY_ORDER = ["Blunder", "Mistake", "Inaccuracy", "Good", "Excellent", "Best"];

export default function AnalysisPanel() {
  const [transcript, setTranscript] = useState("");
  const [data, setData] = useState(null);
  const [sel, setSel] = useState(0);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  async function run(input) {
    setBusy(true);
    setError(null);
    try {
      const d = await api("/analyse", input);
      setData(d);
      setSel(d.positions.length - 1);
    } catch (e) {
      setError(e.message);
      setData(null);
    } finally {
      setBusy(false);
    }
  }

  // deep link: ?analyse=<transcript>
  useEffect(() => {
    const p = new URLSearchParams(location.search);
    const line = p.get("analyse") || p.get("analysis");
    if (line) {
      setTranscript(line);
      run({ transcript: line });
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const graphPoints = useMemo(
    () => (data ? data.eval_graph.map((g) => g.eval_black) : []),
    [data]
  );

  if (!data) {
    return (
      <main>
        <BoardArea grid={emptyGrid()} status="Paste a game to analyse." />
        <section className="panel">
          <Controls
            transcript={transcript}
            setTranscript={setTranscript}
            busy={busy}
            onAnalyse={() => run({ transcript })}
            onUseLast={async () => {
              const st = await api("/state");
              setTranscript((st.history || []).join(" "));
              run({ history_actions: st.history_actions });
            }}
          />
          {error && <p className="error">{error}</p>}
        </section>
      </main>
    );
  }

  const pos = data.positions[sel];
  const ply = data.plies[sel - 1]; // the move that led to `pos`
  const glyphs = {};
  let best = null;
  if (ply) {
    glyphs[ply.played] = { label: ply.label, glyph: ply.glyph };
    if (ply.best !== ply.played) best = ply.best;
    else if (ply.label !== "Best" && ply.label !== "Excellent")
      best = sanToIdx(ply.coach_best_san);
  }

  const status = ply
    ? `Move ${sel}: ${ply.side} played ${ply.played_san} — ${ply.label}. ` +
      `Bot's top: ${ply.top_moves.map((m) => `${m.san} ${Math.round(m.winprob * 100)}%`).join(", ")}`
    : "Starting position.";

  return (
    <main>
      <BoardArea
        grid={pos.grid}
        last={ply ? [ply.played] : []}
        best={best}
        glyphs={glyphs}
        evalBlack={graphPoints[sel]}
        status={status}
      />

      <section className="panel">
        <Controls
          transcript={transcript}
          setTranscript={setTranscript}
          busy={busy}
          onAnalyse={() => run({ transcript })}
          onUseLast={async () => {
            const st = await api("/state");
            setTranscript((st.history || []).join(" "));
            run({ history_actions: st.history_actions });
          }}
        />
        {error && <p className="error">{error}</p>}

        <EvalGraph points={graphPoints} cursor={sel} onSeek={setSel} />

        <div className="analysis-summary">
          {["black", "white"].map((side) => {
            const c = data.summary[side] || {};
            const parts = SUMMARY_ORDER.filter((k) => c[k]);
            return (
              <div key={side}>
                {side === "black" ? "⚫ Black" : "⚪ White"}:{" "}
                {parts.length
                  ? parts.map((k) => (
                      <span key={k}>
                        <b className={"label " + k}>{c[k]}</b> {k}&nbsp;&nbsp;
                      </span>
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
              const s =
                p.best_san !== p.played_san
                  ? p.best_san
                  : p.coach_best_san !== p.played_san
                  ? p.coach_best_san
                  : null;
              if (s) alt = `try ${s} · −${Math.round(p.drop * 100)}`;
            }
            return (
              <li
                key={p.ply}
                className={sel === p.ply + 1 ? "sel" : ""}
                onClick={() => setSel(p.ply + 1)}
              >
                <span className="alt">{p.ply + 1}.</span>
                <span className="mv">
                  {p.side === "black" ? "⚫" : "⚪"}
                  {p.played_san}
                </span>
                <span className="alt">{alt}</span>
                <span className={"label " + p.label}>{p.glyph || p.label}</span>
              </li>
            );
          })}
        </ol>
      </section>
    </main>
  );
}

function Controls({ transcript, setTranscript, busy, onAnalyse, onUseLast }) {
  return (
    <div className="controls">
      <input
        type="text"
        placeholder="paste moves: f5 d6 c3 …  or  f5d6c3…"
        value={transcript}
        onChange={(e) => setTranscript(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && onAnalyse()}
      />
      <button className="primary" onClick={onAnalyse} disabled={busy}>
        {busy ? "Analysing…" : "Analyse"}
      </button>
      <button onClick={onUseLast} disabled={busy}>Use last game</button>
    </div>
  );
}

function emptyGrid() {
  const g = Array.from({ length: 8 }, () => Array(8).fill(0));
  g[3][3] = -1; g[3][4] = 1; g[4][3] = 1; g[4][4] = -1;
  return g;
}

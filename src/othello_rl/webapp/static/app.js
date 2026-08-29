"use strict";
const $ = (s) => document.querySelector(s);
const boardEl = $("#board");

async function api(path, body) {
  const opt = body === undefined
    ? {}
    : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const r = await fetch("/api" + path, opt);
  const data = await r.json();
  if (!r.ok) throw new Error(data.error || r.statusText);
  return data;
}

/* ---------- board rendering ---------- */
function cellIndex(r, c) { return r * 8 + c; }

function renderBoard(grid, opts = {}) {
  const legal = new Set(opts.legal || []);
  const last = new Set(opts.last || []);
  const best = opts.best;
  const glyphs = opts.glyphs || {};
  boardEl.innerHTML = "";
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const i = cellIndex(r, c);
      const cell = document.createElement("div");
      cell.className = "cell";
      const v = grid[r][c];
      if (v !== 0) {
        const d = document.createElement("div");
        d.className = "disc " + (v === 1 ? "black" : "white");
        cell.appendChild(d);
      }
      if (legal.has(i)) cell.classList.add("legal");
      if (last.has(i)) cell.classList.add("last");
      if (best === i) cell.classList.add("bestmove");
      if (glyphs[i] && glyphs[i].glyph) {
        const g = document.createElement("span");
        g.className = "glyph label " + glyphs[i].label;
        g.textContent = glyphs[i].glyph;
        cell.appendChild(g);
      }
      if (legal.has(i) && opts.onMove) cell.onclick = () => opts.onMove(i);
      boardEl.appendChild(cell);
    }
  }
}

function renderEvalBar(winprobBlack) {
  const pct = Math.round((winprobBlack ?? 0.5) * 100);
  $("#evalbar-fill").style.height = pct + "%";
  $("#evalbar-num").textContent = pct + "%";
}

/* ---------- bot badge ---------- */
async function refreshBot() {
  const b = await api("/bot");
  $("#bot-badge").textContent = `bot v${b.version} · ${b.games_finetuned} games fine-tuned`;
  $("#bot-detail").textContent =
    `${b.network.channels}ch×${b.network.blocks} · ${b.params.toLocaleString()} params · ${b.train_env_steps.toLocaleString()} training steps`;
}

/* ================= PLAY ================= */
const Play = {
  state: null,
  async newGame() {
    this.state = await api("/new", { human_color: $("#human-color").value });
    $("#gameover").classList.add("hidden");
    $("#ft-result").classList.add("hidden");
    this.render();
  },
  async move(action) {
    if (!this.state.your_turn) return;
    lockBoard(true);
    try { this.state = await api("/move", { action }); }
    catch (e) { $("#status").textContent = e.message; }
    finally { lockBoard(false); }
    this.render();
  },
  render() {
    const s = this.state;
    if (!s.game_over && s.must_pass && s.your_turn) { this.autoPass(); return; }
    const lastMoves = (s.last_bot_moves || []).map(sanToIdx).filter((x) => x >= 0);
    renderBoard(s.grid, {
      legal: s.your_turn ? s.legal_actions : [],
      last: lastMoves,
      onMove: (i) => this.move(i),
    });
    let msg;
    if (s.game_over) msg = s.winner === "draw" ? "Draw." :
      `${cap(s.winner)} wins ${s.score.black}–${s.score.white}.`;
    else msg = s.your_turn ? "Your move." : "Bot thinking…";
    $("#status").textContent =
      `${msg}   ·   ⚫ ${s.score.black}  ⚪ ${s.score.white}   ·   you are ${cap(s.human_color)}`;
    this.renderMoves(s);
    if (s.game_over) {
      $("#gameover").classList.remove("hidden");
      $("#gameover-text").textContent =
        (s.winner === "draw" ? "Draw" :
          (s.winner === s.human_color ? "You won! " : "The bot won. ")) +
        `Final ${s.score.black}–${s.score.white}.`;
    }
    this.evalCurrent();
  },
  async autoPass() {
    this.state = await api("/move", { action: 64 });
    this.render();
  },
  async evalCurrent() {
    // ask analysis of current game for a live eval bar
    try {
      const a = await api("/analyse", { history_actions: this.state.history_actions });
      const g = a.eval_graph;
      renderEvalBar(g.length ? g[g.length - 1].eval_black : 0.5);
    } catch (_) {}
  },
  renderMoves(s) {
    const ol = $("#move-list"); ol.innerHTML = "";
    s.history.forEach((mv, i) => {
      const li = document.createElement("li");
      const who = i % 2 === 0 ? "⚫" : "⚪";
      li.innerHTML = `${who} <b>${mv}</b>`;
      ol.appendChild(li);
    });
  },
  async finetune() {
    const btn = $("#finetune-btn");
    btn.disabled = true; btn.textContent = "Fine-tuning… (≈10–20s)";
    try {
      const rep = await api("/finetune", {});
      renderFineTune(rep);
      await refreshBot();
    } catch (e) {
      $("#ft-result").classList.remove("hidden");
      $("#ft-result").textContent = "Fine-tune failed: " + e.message;
    } finally {
      btn.disabled = false; btn.textContent = "Fine-tune bot from this game";
    }
  },
};

function renderFineTune(rep) {
  const el = $("#ft-result"); el.classList.remove("hidden");
  const grades = rep.grades || [];
  const counts = {};
  grades.forEach((g) => counts[g.label] = (counts[g.label] || 0) + 1);
  const chips = Object.entries(counts)
    .map(([k, v]) => `<span class="label ${k}">${v} ${k}</span>`).join("  ");
  el.innerHTML = `
    <table>
      <tr><td>bot version</td><td>v${rep.version}${rep.rolled_back ? " (rolled back — update made it weaker)" : ""}</td></tr>
      <tr><td>moves reinforced / penalised</td><td>${rep.n_reinforced} / ${rep.n_penalised}</td></tr>
      <tr><td>TD loss (before → after)</td><td>${rep.loss_before.toFixed(4)} → ${rep.loss_after.toFixed(4)}</td></tr>
      <tr><td>win rate vs Random (guardrail)</td><td>${(rep.winrate_vs_random_before*100).toFixed(0)}% → ${(rep.winrate_vs_random_after*100).toFixed(0)}%</td></tr>
    </table>
    <p style="margin:8px 0 0">bot move grades: ${chips || "—"}</p>`;
}

/* ================= ANALYSIS ================= */
const Analysis = {
  data: null,
  sel: 0,
  async run(input) {
    $("#analyse-btn").disabled = true;
    try {
      this.data = await api("/analyse", input);
      this.sel = this.data.positions.length - 1;
      this.renderGraph();
      this.renderList();
      this.select(this.sel);
    } catch (e) {
      $("#analysis-summary").textContent = e.message;
    } finally { $("#analyse-btn").disabled = false; }
  },
  renderGraph() {
    const g = this.data.eval_graph;
    const svg = $("#eval-graph"); svg.innerHTML = "";
    const W = 600, H = 120;
    const n = g.length;
    const x = (i) => (n <= 1 ? 0 : (i / (n - 1)) * W);
    const y = (e) => H - e * H;
    let d = `M0 ${y(g[0].eval_black)}`;
    g.forEach((p, i) => { d += ` L${x(i).toFixed(1)} ${y(p.eval_black).toFixed(1)}`; });
    const area = `${d} L${W} ${H} L0 ${H} Z`;
    svg.insertAdjacentHTML("beforeend",
      `<line class="mid" x1="0" y1="${H/2}" x2="${W}" y2="${H/2}"/>` +
      `<path class="area" d="${area}"/><path class="line" d="${d}"/>` +
      `<line id="gcursor" class="cursor" x1="0" y1="0" x2="0" y2="${H}"/>`);
    svg.onclick = (ev) => {
      const rect = svg.getBoundingClientRect();
      const i = Math.round(((ev.clientX - rect.left) / rect.width) * (n - 1));
      this.select(Math.max(0, Math.min(n - 1, i)));
    };
  },
  renderList() {
    const ol = $("#analysis-list"); ol.innerHTML = "";
    const s = this.data.summary;
    const ORDER = ["Blunder", "Mistake", "Inaccuracy", "Good", "Excellent", "Best"];
    $("#analysis-summary").innerHTML = ["black", "white"].map((side) => {
      const c = s[side] || {};
      const parts = ORDER.filter((k) => c[k])
        .map((k) => `<b class="label ${k}">${c[k]}</b> ${k}`).join("  ");
      return `<div>${side === "black" ? "⚫ Black" : "⚪ White"}: ${parts || "—"}</div>`;
    }).join("");
    this.data.plies.forEach((p) => {
      const li = document.createElement("li");
      li.dataset.pos = p.ply + 1;
      let alt = "";
      if (p.label !== "Best" && p.label !== "Excellent") {
        const suggestion = p.best_san !== p.played_san ? p.best_san
          : (p.coach_best_san !== p.played_san ? p.coach_best_san : null);
        if (suggestion) alt = `try ${suggestion}  ·  −${Math.round(p.drop * 100)}`;
      }
      li.innerHTML =
        `<span class="alt">${p.ply + 1}.</span>` +
        `<span class="mv">${p.side === "black" ? "⚫" : "⚪"}${p.played_san}</span>` +
        `<span class="alt">${alt}</span>` +
        `<span class="label ${p.label}">${p.glyph || p.label}</span>`;
      li.onclick = () => this.select(p.ply + 1);
      ol.appendChild(li);
    });
  },
  select(posIdx) {
    this.sel = posIdx;
    const pos = this.data.positions[posIdx];
    const ply = this.data.plies[posIdx - 1]; // the move that led here
    const glyphs = {};
    let best;
    if (ply) {
      glyphs[ply.played] = { label: ply.label, glyph: ply.glyph };
      if (ply.best !== ply.played) best = ply.best;
      else if (ply.label !== "Best" && ply.label !== "Excellent") best = sanToIdx(ply.coach_best_san);
    }
    renderBoard(pos.grid, { last: ply ? [ply.played] : [], best, glyphs });
    const g = this.data.eval_graph[posIdx] || this.data.eval_graph[this.data.eval_graph.length - 1];
    renderEvalBar(g.eval_black);
    const cur = $("#gcursor");
    if (cur) { const xx = (posIdx / (this.data.positions.length - 1)) * 600; cur.setAttribute("x1", xx); cur.setAttribute("x2", xx); }
    document.querySelectorAll(".analysis-list li").forEach((li) =>
      li.classList.toggle("sel", +li.dataset.pos === posIdx));
    if (ply) {
      $("#status").textContent =
        `Move ${posIdx}: ${ply.side} played ${ply.played_san} — ${ply.label}. ` +
        `Bot's top: ${ply.top_moves.map((m) => `${m.san} ${Math.round(m.winprob*100)}%`).join(", ")}`;
    } else {
      $("#status").textContent = "Starting position.";
    }
  },
};

/* ---------- helpers / wiring ---------- */
function cap(s) { return s ? s[0].toUpperCase() + s.slice(1) : s; }
function lockBoard(on) { boardEl.style.pointerEvents = on ? "none" : ""; }
function sanToIdx(san) {
  if (!san || san === "pass") return -1;
  const c = san.charCodeAt(0) - 97, r = parseInt(san[1], 10) - 1;
  return (r >= 0 && r < 8 && c >= 0 && c < 8) ? r * 8 + c : -1;
}

function switchTab(name) {
  document.querySelectorAll(".tabs button").forEach((b) =>
    b.classList.toggle("active", b.dataset.tab === name));
  $("#panel-play").classList.toggle("hidden", name !== "play");
  $("#panel-analysis").classList.toggle("hidden", name !== "analysis");
  if (name === "play" && Play.state) Play.render();
}

document.querySelectorAll(".tabs button").forEach((b) =>
  b.onclick = () => switchTab(b.dataset.tab));
$("#new-game").onclick = () => Play.newGame();
$("#finetune-btn").onclick = () => Play.finetune();
$("#analyse-btn").onclick = () => Analysis.run({ transcript: $("#transcript").value });
$("#analyse-current").onclick = () => {
  if (Play.state) Analysis.run({ history_actions: Play.state.history_actions });
};
$("#reset-bot").onclick = async () => {
  if (!confirm("Reset the bot to its baseline weights? This discards all fine-tuning.")) return;
  await api("/bot/reset", {});
  await refreshBot();
};

/* deep links:  ?analyse=f5d6c3   or   #analysis   (Lichess-style shareable analysis) */
(async () => {
  await refreshBot();
  await Play.newGame();
  const params = new URLSearchParams(location.search);
  const line = params.get("analyse") || params.get("analysis");
  if (line) {
    switchTab("analysis");
    $("#transcript").value = line;
    await Analysis.run({ transcript: line });
  } else if (location.hash === "#analysis") {
    switchTab("analysis");
  }
})();

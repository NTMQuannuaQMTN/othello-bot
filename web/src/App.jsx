import { useCallback, useEffect, useState } from "react";
import PlayPanel from "./components/PlayPanel.jsx";
import AnalysisPanel from "./components/AnalysisPanel.jsx";
import BotBadge from "./components/BotBadge.jsx";
import { api } from "./api.js";

const apiBase = () => (import.meta.env.VITE_API_BASE ?? "").replace(/\/$/, "") || location.origin;
const isLocal = () => ["localhost", "127.0.0.1"].includes(location.hostname);

export default function App() {
  const [tab, setTab] = useState(
    location.hash === "#analysis" || new URLSearchParams(location.search).has("analyse")
      ? "analysis"
      : "play"
  );
  const [bot, setBot] = useState(null);
  const [apiDown, setApiDown] = useState(false);
  const [analyzeLine, setAnalyzeLine] = useState(null); // actions[] to load in Analysis

  const analyzeGame = useCallback((actions) => {
    setAnalyzeLine({ actions, nonce: Date.now() });
    setTab("analysis");
  }, []);

  const [apiErr, setApiErr] = useState(null);
  const refreshBot = useCallback(async () => {
    try {
      setBot(await api("/bot"));
      setApiDown(false);
      setApiErr(null);
    } catch (e) {
      setApiDown(true);
      setApiErr(e?.message || String(e));
    }
  }, []);

  useEffect(() => {
    refreshBot();
    const id = setInterval(refreshBot, 3000); // keep retrying while the API is down
    return () => clearInterval(id);
  }, [refreshBot]);

  return (
    <>
      <header>
        <h1>Othello<span className="accent">RL</span></h1>
        <nav className="tabs">
          <button className={tab === "play" ? "active" : ""} onClick={() => setTab("play")}>
            Play
          </button>
          <button
            className={tab === "analysis" ? "active" : ""}
            onClick={() => setTab("analysis")}
          >
            Analysis
          </button>
        </nav>
        <BotBadge bot={bot} />
      </header>

      {apiDown && (
        <div className="api-down">
          Can't reach the bot API at <code>{apiBase()}/api/bot</code>.
          {isLocal()
            ? <> Start it in another terminal: <code>python3 scripts/serve.py</code>.</>
            : <> The server may be starting up or misconfigured
                — try again in a moment.</>}
          {apiErr && <div className="alt" style={{ marginTop: 6 }}>{apiErr}</div>}
        </div>
      )}

      {tab === "play" ? (
        <PlayPanel onAnalyzeGame={analyzeGame} />
      ) : (
        <AnalysisPanel loadLine={analyzeLine} />
      )}

      <footer>
        <span>
          <code>
            {bot
              ? `${bot.network.channels}ch×${bot.network.blocks} DQN + search · ${bot.params.toLocaleString()} params`
              : "…"}
          </code>
        </span>
      </footer>
    </>
  );
}

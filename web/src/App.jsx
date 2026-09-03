import { useCallback, useEffect, useState } from "react";
import PlayPanel from "./components/PlayPanel.jsx";
import AnalysisPanel from "./components/AnalysisPanel.jsx";
import BotBadge from "./components/BotBadge.jsx";
import { api } from "./api.js";

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

  const refreshBot = useCallback(async () => {
    try {
      setBot(await api("/bot"));
      setApiDown(false);
    } catch {
      setApiDown(true);
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
          Can't reach the bot API. If you're running it yourself, start it with{" "}
          <code>python3 scripts/serve.py</code>.
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

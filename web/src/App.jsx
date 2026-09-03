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

  async function resetBot() {
    if (!confirm("Reset the bot to its baseline weights? This discards all fine-tuning."))
      return;
    await api("/bot/reset", {});
    refreshBot();
  }

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
          Can't reach the bot API on <code>:8000</code>. Start it in another
          terminal:&nbsp;
          <code>python3 scripts/serve.py --config configs/webapp.yaml</code>
          &nbsp;(or run <code>npm run dev:all</code> instead of <code>npm run dev</code>).
        </div>
      )}

      {tab === "play" ? (
        <PlayPanel onBotChanged={refreshBot} onAnalyzeGame={analyzeGame}
          canFinetune={bot?.features?.finetune ?? true} />
      ) : (
        <AnalysisPanel loadLine={analyzeLine} onBotChanged={refreshBot}
          canFinetune={bot?.features?.finetune ?? true} />
      )}

      <footer>
        <span>
          DQN bot ·{" "}
          <code>
            {bot
              ? `${bot.network.channels}ch×${bot.network.blocks} · ${bot.params.toLocaleString()} params · ${bot.train_env_steps.toLocaleString()} training steps`
              : "…"}
          </code>
        </span>
        {(bot?.features?.finetune ?? true) && (
          <button className="link" onClick={resetBot}>reset bot to baseline</button>
        )}
      </footer>
    </>
  );
}

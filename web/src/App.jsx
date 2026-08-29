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

  const refreshBot = useCallback(async () => {
    try {
      setBot(await api("/bot"));
    } catch {
      /* ignore */
    }
  }, []);

  useEffect(() => { refreshBot(); }, [refreshBot]);

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

      {tab === "play" ? (
        <PlayPanel onBotChanged={refreshBot} />
      ) : (
        <AnalysisPanel />
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
        <button className="link" onClick={resetBot}>reset bot to baseline</button>
      </footer>
    </>
  );
}

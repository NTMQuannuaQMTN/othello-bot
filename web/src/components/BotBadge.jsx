export default function BotBadge({ bot }) {
  if (!bot) return <div className="badge">bot …</div>;
  return (
    <div className="badge" title="the bot the site plays and analyses with">
      bot v{bot.version} · {bot.games_finetuned} games fine-tuned
    </div>
  );
}

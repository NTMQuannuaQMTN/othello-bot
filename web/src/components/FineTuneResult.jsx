const ORDER = ["Blunder", "Mistake", "Inaccuracy", "Good", "Excellent", "Best"];

export default function FineTuneResult({ report, error }) {
  if (error) return <div className="ft-result error">Fine-tune failed: {error}</div>;
  if (!report) return null;

  const counts = {};
  for (const g of report.grades || []) counts[g.label] = (counts[g.label] || 0) + 1;

  return (
    <div className="ft-result">
      <table>
        <tbody>
          <tr>
            <td>bot version</td>
            <td>
              v{report.version}
              {report.rolled_back && " — rolled back (update made it weaker)"}
            </td>
          </tr>
          <tr>
            <td>moves reinforced / penalised</td>
            <td>{report.n_reinforced} / {report.n_penalised}</td>
          </tr>
          <tr>
            <td>TD loss (before → after)</td>
            <td>{report.loss_before.toFixed(4)} → {report.loss_after.toFixed(4)}</td>
          </tr>
          <tr>
            <td>win rate vs Random (guardrail)</td>
            <td>
              {(report.winrate_vs_random_before * 100).toFixed(0)}% →{" "}
              {(report.winrate_vs_random_after * 100).toFixed(0)}%
            </td>
          </tr>
        </tbody>
      </table>
      <div className="chips">
        {ORDER.filter((k) => counts[k]).map((k) => (
          <span key={k} className={"label " + k}>{counts[k]} {k}</span>
        ))}
      </div>
    </div>
  );
}

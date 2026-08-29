export default function EvalGraph({ points, cursor, onSeek }) {
  const W = 600, H = 120;
  const n = points.length;
  if (n < 2) return <svg className="eval-graph" viewBox={`0 0 ${W} ${H}`} />;
  const x = (i) => (i / (n - 1)) * W;
  const y = (e) => H - e * H;
  let d = `M0 ${y(points[0])}`;
  points.forEach((p, i) => { d += ` L${x(i).toFixed(1)} ${y(p).toFixed(1)}`; });
  const area = `${d} L${W} ${H} L0 ${H} Z`;
  const cx = cursor != null ? x(cursor) : null;

  const handleClick = (ev) => {
    if (!onSeek) return;
    const rect = ev.currentTarget.getBoundingClientRect();
    const i = Math.round(((ev.clientX - rect.left) / rect.width) * (n - 1));
    onSeek(Math.max(0, Math.min(n - 1, i)));
  };

  return (
    <svg className="eval-graph" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" onClick={handleClick}>
      <line className="mid" x1="0" y1={H / 2} x2={W} y2={H / 2} />
      <path className="area" d={area} />
      <path className="line" d={d} />
      {cx != null && <line className="cursor" x1={cx} y1="0" x2={cx} y2={H} />}
    </svg>
  );
}

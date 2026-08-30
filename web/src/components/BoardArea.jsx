import Board from "./Board.jsx";

export default function BoardArea({ evalBlack = 0.5, status, footer, ...boardProps }) {
  const pct = Math.round((evalBlack ?? 0.5) * 100);
  const boardHeight = 8 * 54 + 12;
  return (
    <div className="board-area">
      <div className="board-row">
        <div className="evalbar" style={{ height: boardHeight }}>
          <div style={{ height: pct + "%" }} />
          <span>{pct}</span>
        </div>
        <Board {...boardProps} />
      </div>
      <div className="status">{status}</div>
      {footer}
    </div>
  );
}

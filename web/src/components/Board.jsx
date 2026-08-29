export default function Board({ grid, legal = [], last = [], best = null, glyphs = {}, onMove }) {
  const legalSet = new Set(legal);
  const lastSet = new Set(last);
  const cells = [];
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const i = r * 8 + c;
      const v = grid[r][c];
      const isLegal = legalSet.has(i);
      const cls = [
        "cell",
        isLegal && "legal",
        lastSet.has(i) && "last",
        best === i && "bestmove",
      ].filter(Boolean).join(" ");
      const g = glyphs[i];
      cells.push(
        <div
          key={i}
          className={cls}
          onClick={isLegal && onMove ? () => onMove(i) : undefined}
        >
          {v !== 0 && <div className={"disc " + (v === 1 ? "black" : "white")} />}
          {g && g.glyph && <span className={"glyph label " + g.label}>{g.glyph}</span>}
        </div>
      );
    }
  }
  return <div className="board">{cells}</div>;
}

import { useEffect, useRef } from "react";

/**
 * Shared move list for Play and Analysis. `items` is
 *   [{ n, side, san, note?, right?, rightClass? }]
 * `selected` is a 1-based ply index (0 = start / none). Clicking row i calls
 * `onSelect(i + 1)`.
 */
export default function MoveList({ items, selected = 0, onSelect,
                                   emptyText = "no moves yet" }) {
  const listRef = useRef(null);
  const selRef = useRef(null);
  const endRef = useRef(null);

  // keep the current move visible as you navigate (and follow a live game)
  useEffect(() => {
    const target = selRef.current
      || (selected >= items.length ? endRef.current : null);
    target?.scrollIntoView({ block: "nearest" });
  }, [selected, items.length]);

  return (
    <ol className="move-list-shared" ref={listRef}>
      {items.length === 0 && <li className="alt">{emptyText}</li>}
      {items.map((m, i) => (
        <li key={i}
          ref={selected === i + 1 ? selRef : null}
          className={selected === i + 1 ? "sel" : ""}
          onClick={onSelect ? () => onSelect(i + 1) : undefined}>
          <span className="n">{m.n ?? i + 1}.</span>
          <span className="mv">
            {m.side === "black" ? "⚫" : "⚪"}{m.san}
          </span>
          <span className="note">{m.note || ""}</span>
          {m.right != null && (
            <span className={"right " + (m.rightClass || "")}>{m.right}</span>
          )}
        </li>
      ))}
      <li ref={endRef} aria-hidden className="mv-end" />
    </ol>
  );
}

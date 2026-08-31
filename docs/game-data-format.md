# Historical game data — formats & ingestion

The historical-training pipeline (`docs/historical-training.md`) starts by turning
raw game files into one internal representation: **`GameRecord`**
(`src/othello_rl/ingest/records.py`). Ingestion *only parses* — legality and
replay validation happen in the next stage (`scripts/validate_games.py`).

## `GameRecord`

| field | meaning |
|---|---|
| `source` | `"wthor"`, `"webapp"`, `"transcript"`, `"generic"`, … |
| `source_format` | `"wtb"`, `"jsonl"`, `"transcript"`, `"json"` |
| `moves` | flat action indices `row*8+col` (0–63), `64` = pass, **as the source stores them** — most sources omit forced passes |
| `game_id` | stable id; defaults to `"<source>:<sha1(move_signature)[:16]>"` |
| `data_kind` | `historical` \| `self_play` \| `engine_generated` — kept on every downstream training example |
| `metadata` | free dict: `year`, `tournament`, `black_player`, ratings, … |
| `result` | `{black_discs, white_discs, winner}` or `null` when unknown |
| `provenance` | `file`, `record_index`/`line`, `pass_convention` (`implicit`\|`explicit`) |
| `ingested_at` | timestamp |
| `canonical_moves` | filled by validation: `moves` with forced passes inserted |

`move_signature()` (placements only, passes stripped) is the **de-duplication
key** — the same game from two databases collapses to one.

JSON round-trips via `GameRecord.to_json()` / `.from_json()`. Files are JSONL
(one record per line) under `data/processed/validated_games/<source>.raw.jsonl`.

## Sources

### WThor database (`.wtb`) — primary historical source

Published by the **Fédération Française d'Othello** at
<http://www.ffothello.org/informatique/la-base-wthor/>. The `.wtb` files are a
free static download — no login, API key, rate limit or anti-bot protection. This
project **does not download them**; place them yourself:

```
data/raw/wthor/
├── WTH_2004.wtb        (one file per year, ~1990–present)
├── WTH_2005.wtb
├── …
├── WTHOR.JOU           (optional: player-name table)
└── WTHOR.TRN           (optional: tournament-name table)
```

Then `python3 scripts/ingest_games.py --source wthor`.

**`.wtb` layout** (little-endian), parsed by
`src/othello_rl/ingest/sources/wthor.py`:

```
header  16 bytes
  0..3    creation century / year / month / day
  4..7    N1  number of games            (uint32)
  8..9    N2  number of records          (uint16)
  10..11  game year                      (uint16)
  12..15  board size / kind / depth / reserved

then N1 game records, 68 bytes each
  0..1   tournament label number   (uint16)
  2..3   black player number       (uint16)
  4..5   white player number       (uint16)
  6      real score  = black disc count at game end (0..64)
  7      theoretical score (with perfect play)
  8..67  60 move bytes, each = 10*row + col (row, col in 1..8); 0 = game ended
```

Only 8×8 games are supported (the record must be 68 bytes). Forced passes are not
stored — the validator inserts them. Attribution: cite the FFO WThor database in
any published results.

### Our own web-app games (`--source webapp` / `jsonl`)

`data/games.jsonl` lines: `{"moves":[…], "winner":"…", "score":{"black":n,"white":n}, …}`.
`moves` already contains forced passes (`pass_convention: explicit`).

### Plain transcripts (`--source transcript`)

`.txt` / `.gam` files, one game per line (or one game per file): `f5 d6 c3 …`,
`f5,d6`, or run-together `f5d6c3…`. `pass` allowed. `#` lines are comments.

### Generic JSON (`--source generic`)

`[ {game}, … ]` or `{"games": [ {game}, … ]}` where a game is
`{"moves": [...], "id"?, "metadata"?, "result"?, "data_kind"?}` and a move is an
int action or a square name.

## De-duplication

`scripts/ingest_games.py` de-duplicates by `move_signature()` across every file
and every source in one run (pass `--no-dedup` to keep everything). The
`.ingest.json` sidecar records `parsed / duplicates / written`.

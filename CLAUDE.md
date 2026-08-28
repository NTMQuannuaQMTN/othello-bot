# Claude Code / repo notes

- Only Python 3.9.6 on this machine; deps are `pip install --user` in ~/Library/Python/3.9. Run tests with: `export PATH="$HOME/Library/Python/3.9/bin:$PATH"; python3 -m pytest` (or `PYTHONPATH=src python3 -m pytest`).
- Code targets Python 3.9+ (no match statements, no `X | Y` runtime unions; `from __future__ import annotations` in modules using modern hints).
- Board: int8 (8,8), +1 black / -1 white / 0 empty. Black first. See PROJECT_SPEC.md.
- Workflow: implement smallest increment, add tests, `python3 -m pytest`, update TASKS.md + PROGRESS.md, commit.

# AgenticKapruka — Agent Instructions

## Project

Agentic shopping assistant for Kapruka: FastAPI + HTMX + LangGraph + Neo4j GraphRAG,
deployed to Google Cloud Run. Kapruka product data flows through the Kapruka MCP server.

## Quality Bar

Production code. Maintainable, tested, typed. Run before every commit:

- `ruff check .` and `ruff format --check .`
- `mypy app/ lib/ graphs/`
- `pytest tests/unit -q`

## Conventions

- Python 3.12+, Pydantic v2, async-first
- FastAPI app factory in `app/main.py`
- Business logic in `lib/`, LangGraph in `graphs/`
- Jinja2 templates in `templates/`, static assets in `static/`
- One logical change per commit; conventional commit messages `feat(PRD-XXX): title`

## Kapruka MCP

Read-only tools may be called live for verification. Do not call `kapruka_create_order`
until checkout PRDs (PRD-070+). Respect rate limits — use MCP sparingly before PRD-009 ships.

## Ralph Workflow

Work one PRD item at a time from `prd.json`. Update `passes: true` when done.
Append decisions to `progress.txt`. Commit on `ralph/sprint-1` branch.

```bash
./scripts/ralph-once.sh          # single supervised iteration (headless JSON)
./scripts/ralph-once.sh -i       # same PRD prompt in interactive Cursor Agent UI
./scripts/ralph.sh [N]           # AFK loop (default 10 iterations)
```

# AgenticKapruka

Agentic shopping assistant for [Kapruka](https://www.kapruka.com): FastAPI + HTMX + LangGraph + Neo4j GraphRAG, deployed to Google Cloud Run.

## Stack

- **API / UI**: FastAPI, Jinja2, HTMX, Alpine.js, Tailwind CSS
- **Agent**: LangGraph orchestration, Google GenAI / Vertex
- **Data**: Neo4j GraphRAG, Redis (sessions/cart), Zep (memory)
- **Integration**: Kapruka MCP server for live product and order data

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

Copy `.env.example` to `.env` once PRD-004 lands, then:

```bash
uvicorn app.main:app --reload
```

## Development

Run quality checks before every commit:

```bash
ruff check .
ruff format --check .
mypy app/ lib/ graphs/
pytest tests/unit -q
```

See [AGENTS.md](AGENTS.md) for agent and contributor conventions.

## Ralph autonomous workflow

PRD backlog lives in `prd.json`. Progress is logged in `progress.txt`.

```bash
# Single supervised iteration (headless JSON stream — good for logs)
./scripts/ralph-once.sh

# Interactive Cursor Agent UI (full-screen TUI)
./scripts/ralph-once.sh --interactive
# or: RALPH_INTERACTIVE=1 ./scripts/ralph-once.sh

# Headless plain text instead of JSON
./scripts/ralph-once.sh --text

# AFK loop (default 10 iterations, headless JSON stream)
./scripts/ralph.sh

# AFK loop with readable plain text instead of JSON
./scripts/ralph.sh 10 --text

# Custom iteration count and model
./scripts/ralph.sh 5 --model composer-2.5
```

Work happens on branch `ralph/sprint-1`. One PRD item per commit: `feat(PRD-XXX): title`.

## Project layout

```
app/          FastAPI routes, config, templating
lib/          Business logic and service clients
graphs/       LangGraph agent definitions
templates/    Jinja2 HTML partials
static/       Compiled CSS and assets
tests/        Unit and integration tests
scripts/      Ralph loop and tooling
.cursor/      Cursor agent skills (Ralph personas)
```

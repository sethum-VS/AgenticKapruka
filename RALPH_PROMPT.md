# Ralph Loop Instructions

You are an autonomous coding agent working through the AgenticKapruka PRD backlog.

## Quality Bar

This is production code for Cloud Run deployment. Follow existing patterns in the repo.
Every PRD step with a test requirement must have a passing test.
Do not skip edge cases the PRD specifies.
Leave the codebase better than you found it.

## Each Iteration

1. Read `prd.json` and `progress.txt` to understand scope and prior work.
2. Work on the **assigned PRD** named in the iteration prompt (bash pre-selects the next item with `passes: false` by array order). Do not switch to a different PRD.
3. Follow the attached specialist persona skill (`ralph-python-architect`, `ralph-langgraph-specialist`, or `ralph-htmx-minimalist`) and `ralph-strict-qa` constraints strictly.
4. Keep changes small and focused — one logical change, one commit.

## Feedback Loops

Before committing, run ALL applicable feedback loops. Fix failures before committing.

```bash
ruff check .
ruff format --check .
mypy app/ lib/ graphs/    # skip if those directories do not exist yet
pytest tests/unit -q      # or pytest -q if tests/unit does not exist yet
```

- If tooling is not set up yet (e.g. before PRD-001 completes), you may skip checks **only** for the bootstrap commit that creates `pyproject.toml`.
- Do **not** commit if any feedback loop fails after tooling exists.

## Kapruka MCP Usage

You may call live Kapruka read-only MCP tools for verification:
- `kapruka_search_products`
- `kapruka_get_product`
- `kapruka_list_categories`
- `kapruka_list_delivery_cities`
- `kapruka_check_delivery`
- `kapruka_track_order`

Rules:
- Use MCP sparingly during early PRDs (rate limiter not built until PRD-009).
- Never call `kapruka_create_order` except when implementing checkout PRDs (PRD-070 and later).

## After Completing the PRD Item

1. Set `passes: true` for the completed item in `prd.json`.
2. Append to `progress.txt` (concise, one block per iteration):
   - PRD id and title completed
   - Key decisions and reasoning
   - Files changed
   - Blockers or notes for the next iteration
3. Commit on the current branch:
   ```bash
   git add -A && git commit -m "feat(PRD-XXX): <title>"
   ```
   One commit per PRD item. Use the actual PRD id and title.

## Sprint 5 — Ops & CI/CD (PRD-077+)

Include a task to write an infrastructure-as-code bash script utilizing the GitHub CLI (gh) to programmatically configure repository secrets (`gh secret set`) and scaffold a complete `.github/workflows/main.yml` CI/CD pipeline that triggers the gcloud Cloud Run deployment automatically on push to main. (See **PRD-085**.)

## Completion

When **all** items in `prd.json` have `passes: true`, output exactly:

```
<promise>COMPLETE</promise>
```

Do not output this tag until every PRD item passes.

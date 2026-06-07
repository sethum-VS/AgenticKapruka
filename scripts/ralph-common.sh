#!/usr/bin/env bash
# Shared functions for Ralph Wiggum loop scripts (ralph-once.sh, ralph.sh)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RALPH_BRANCH="${RALPH_BRANCH:-ralph/sprint-1}"
RALPH_MODEL="${RALPH_MODEL:-composer-2.5}"
RALPH_SANDBOX="${RALPH_SANDBOX:-}"
RALPH_STREAM="${RALPH_STREAM:-1}"
RALPH_INTERACTIVE="${RALPH_INTERACTIVE:-}"
RALPH_LAST_LOG=""

PRD_FILE="$PROJECT_ROOT/prd.json"
PROGRESS_FILE="$PROJECT_ROOT/progress.txt"
PROMPT_FILE="$PROJECT_ROOT/RALPH_PROMPT.md"
AGENTS_FILE="$PROJECT_ROOT/AGENTS.md"

require_jq() {
  if ! command -v jq >/dev/null 2>&1; then
    echo "Error: jq is required for Ralph (prd.json). Install: brew install jq" >&2
    exit 1
  fi
}

warn_unknown_args() {
  if [[ ${#REMAINING_ARGS[@]} -gt 0 ]]; then
    echo "Warning: ignoring unknown arguments: ${REMAINING_ARGS[*]}" >&2
  fi
}

cursor_agent_bin() {
  if command -v cursor-agent >/dev/null 2>&1; then
    command -v cursor-agent
  elif [[ -x "${HOME}/.local/bin/cursor-agent" ]]; then
    echo "${HOME}/.local/bin/cursor-agent"
  else
    echo "cursor-agent not found. Install: curl https://cursor.com/install -fsS | bash" >&2
    return 1
  fi
}

parse_ralph_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --interactive|-i)
        RALPH_INTERACTIVE=1
        RALPH_STREAM=0
        shift
        ;;
      --text|--no-stream)
        RALPH_STREAM=0
        shift
        ;;
      --model)
        if [[ $# -lt 2 || -z "${2:-}" ]]; then
          echo "Error: --model requires a value." >&2
          exit 1
        fi
        RALPH_MODEL="$2"
        shift 2
        ;;
      --model=*)
        RALPH_MODEL="${1#*=}"
        shift
        ;;
      *)
        REMAINING_ARGS+=("$1")
        shift
        ;;
    esac
  done
}

ensure_git_repo() {
  if ! git -C "$PROJECT_ROOT" rev-parse --git-dir >/dev/null 2>&1; then
    echo "Error: $PROJECT_ROOT is not a git repository." >&2
    exit 1
  fi
}

ensure_branch() {
  ensure_git_repo
  cd "$PROJECT_ROOT"
  if git show-ref --verify --quiet "refs/heads/$RALPH_BRANCH"; then
    git checkout "$RALPH_BRANCH"
  else
    git checkout -b "$RALPH_BRANCH"
  fi
}

all_prd_pass() {
  [[ "$(jq '[.[] | select(.passes == false)] | length' "$PRD_FILE")" -eq 0 ]]
}

select_next_prd_id() {
  jq -r '.[] | select(.passes == false) | .id' "$PRD_FILE" | head -1
}

get_prd_field() {
  local id="$1"
  local field="$2"
  jq -r --arg id "$id" --arg field "$field" '.[] | select(.id == $id) | .[$field]' "$PRD_FILE"
}

resolve_specialist_persona() {
  local id="$1"
  local category title description
  category="$(get_prd_field "$id" category)"
  title="$(get_prd_field "$id" title)"
  description="$(get_prd_field "$id" description)"

  case "$category" in
    architecture|ops|mcp)
      echo "python-architect"
      ;;
    orchestration|graphrag)
      echo "langgraph-specialist"
      ;;
    ui)
      echo "htmx-minimalist"
      ;;
    checkout)
      if echo "$title $description" | grep -qiE 'graph|sub-graph|checkout_graph|node'; then
        echo "langgraph-specialist"
      elif echo "$title $description" | grep -qiE 'redis|server-side cart'; then
        echo "python-architect"
      else
        echo "htmx-minimalist"
      fi
      ;;
    *)
      echo "python-architect"
      ;;
  esac
}

build_prompt() {
  if [[ ! -f "$PROMPT_FILE" ]]; then
    echo "Error: $PROMPT_FILE not found." >&2
    exit 1
  fi

  local prd_id specialist persona_skill agents_ref=""
  prd_id="$(select_next_prd_id)"

  if [[ -z "$prd_id" ]]; then
    cat <<EOF
All PRD items have passes: true. Output exactly:

<promise>COMPLETE</promise>
EOF
    return
  fi

  specialist="$(resolve_specialist_persona "$prd_id")"
  persona_skill="ralph-${specialist}"

  local prd_title
  prd_title="$(get_prd_field "$prd_id" title)"

  if [[ -f "$AGENTS_FILE" ]]; then
    agents_ref="@$AGENTS_FILE
"
  fi

  cat <<EOF
@$PROJECT_ROOT/.cursor/skills/ralph-strict-qa/SKILL.md
@$PROJECT_ROOT/.cursor/skills/$persona_skill/SKILL.md
${agents_ref}@$PRD_FILE
@$PROGRESS_FILE

## Assigned task this iteration

Work ONLY on $prd_id ($prd_title). Do not start other PRDs.

$(cat "$PROMPT_FILE")
EOF
}

log_iteration_context() {
  local sandbox_label ui_label prd_id specialist prd_title
  if [[ -n "$RALPH_SANDBOX" ]]; then
    sandbox_label="on"
  else
    sandbox_label="off"
  fi

  if [[ -n "$RALPH_INTERACTIVE" ]]; then
    ui_label="interactive"
  elif [[ "$RALPH_STREAM" == "1" ]]; then
    ui_label="headless-json"
  else
    ui_label="headless-text"
  fi

  prd_id="$(select_next_prd_id)"
  if [[ -z "$prd_id" ]]; then
    echo "branch: $RALPH_BRANCH | model: $RALPH_MODEL | ui: $ui_label | sandbox: $sandbox_label | PRD: (none — all pass)" >&2
    return
  fi

  specialist="$(resolve_specialist_persona "$prd_id")"
  prd_title="$(get_prd_field "$prd_id" title)"
  echo "branch: $RALPH_BRANCH | model: $RALPH_MODEL | ui: $ui_label | sandbox: $sandbox_label | PRD: $prd_id ($prd_title) | personas: strict-qa + $specialist" >&2
}

log_iteration_commit() {
  local before_sha="$1"
  local after_sha
  after_sha="$(git -C "$PROJECT_ROOT" rev-parse HEAD)"
  if [[ "$before_sha" != "$after_sha" ]]; then
    git -C "$PROJECT_ROOT" log -1 --oneline >&2
  else
    echo "no new commit" >&2
  fi
}

_build_agent_args() {
  RALPH_AGENT_ARGS=(
    --force
    --approve-mcps
    --model "$RALPH_MODEL"
    --workspace "$PROJECT_ROOT"
  )

  if [[ -n "$RALPH_INTERACTIVE" ]]; then
    # Full-screen Cursor Agent TUI (no -p / --print).
    :
  else
    RALPH_AGENT_ARGS+=(-p --trust)
    if [[ "$RALPH_STREAM" == "1" ]]; then
      RALPH_AGENT_ARGS+=(--output-format stream-json --stream-partial-output)
    else
      RALPH_AGENT_ARGS+=(--output-format text)
    fi
  fi

  if [[ -n "$RALPH_SANDBOX" ]]; then
    RALPH_AGENT_ARGS+=(--sandbox enabled)
  fi
}

_run_agent_with_optional_timeout() {
  local agent_bin="$1"
  shift
  local -a agent_args=("$@")

  if [[ -n "${RALPH_TIMEOUT:-}" && "$RALPH_TIMEOUT" != "0" ]]; then
    if command -v timeout >/dev/null 2>&1; then
      timeout "$RALPH_TIMEOUT" "$agent_bin" "${agent_args[@]}"
      return $?
    elif command -v gtimeout >/dev/null 2>&1; then
      gtimeout "$RALPH_TIMEOUT" "$agent_bin" "${agent_args[@]}"
      return $?
    else
      echo "Warning: RALPH_TIMEOUT=$RALPH_TIMEOUT set but timeout/gtimeout not found; running without limit." >&2
    fi
  fi

  "$agent_bin" "${agent_args[@]}"
}

run_ralph_iteration() {
  local agent_bin before_sha prompt agent_exit=0 iter_start
  local -a agent_args=()

  agent_bin="$(cursor_agent_bin)"
  log_iteration_context
  before_sha="$(git -C "$PROJECT_ROOT" rev-parse HEAD)"

  if [[ -n "${RALPH_LAST_LOG:-}" && -f "$RALPH_LAST_LOG" ]]; then
    rm -f "$RALPH_LAST_LOG"
  fi
  RALPH_LAST_LOG="$(mktemp "${TMPDIR:-/tmp}/ralph-iter.XXXXXX")"

  _build_agent_args
  agent_args=("${RALPH_AGENT_ARGS[@]}")
  prompt="$(build_prompt)"

  iter_start="$SECONDS"
  if [[ -n "${RALPH_TIMEOUT:-}" && "$RALPH_TIMEOUT" != "0" ]]; then
    echo "Agent timeout: ${RALPH_TIMEOUT}s (unset or RALPH_TIMEOUT=0 to disable)" >&2
  fi
  echo "Agent started at $(date '+%H:%M:%S') — output streams below..." >&2

  if [[ -n "$RALPH_INTERACTIVE" ]]; then
    echo "Launching Cursor Agent interactive UI — quit the session when the PRD item is done." >&2
    _run_agent_with_optional_timeout "$agent_bin" "${agent_args[@]}" "$prompt" | tee "$RALPH_LAST_LOG"
    agent_exit="${PIPESTATUS[0]}"
  else
    # Stream live via tee — do NOT use command substitution (buffers until agent exits).
    _run_agent_with_optional_timeout "$agent_bin" "${agent_args[@]}" "$prompt" | tee "$RALPH_LAST_LOG"
    agent_exit="${PIPESTATUS[0]}"
  fi

  echo "Agent finished in $((SECONDS - iter_start))s (exit $agent_exit)" >&2
  log_iteration_commit "$before_sha"
  return "$agent_exit"
}

is_user_abort_exit() {
  local code="${1:-$?}"
  [[ "$code" -eq 130 || "$code" -eq 143 ]]
}

is_complete() {
  local log_file="${1:-$RALPH_LAST_LOG}"
  if [[ -n "$log_file" && -f "$log_file" ]] && grep -q '<promise>COMPLETE</promise>' "$log_file"; then
    return 0
  fi
  all_prd_pass
}

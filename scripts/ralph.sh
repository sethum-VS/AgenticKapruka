#!/usr/bin/env bash
# AFK Ralph — run up to N autonomous iterations with sandbox enabled.
# Usage:
#   ./scripts/ralph.sh [iterations] [--model MODEL]
#   ./scripts/ralph.sh [iterations] [--text|--no-stream] [--model MODEL]
# Default: 10 iterations, headless JSON stream (RALPH_STREAM=1)

set -euo pipefail

REMAINING_ARGS=()
source "$(cd "$(dirname "$0")" && pwd)/ralph-common.sh"

for arg in "$@"; do
  case "$arg" in
    --interactive|-i)
      echo "Error: --interactive is only supported by ralph-once.sh (single HITL session)." >&2
      echo "Run: ./scripts/ralph-once.sh --interactive" >&2
      exit 1
      ;;
  esac
done

parse_ralph_args "$@"

ITERATIONS="${REMAINING_ARGS[0]:-10}"
if ! [[ "$ITERATIONS" =~ ^[0-9]+$ ]] || [[ "$ITERATIONS" -lt 1 ]]; then
  echo "Usage: $0 [iterations] [--text|--no-stream] [--model MODEL]" >&2
  echo "  iterations must be a positive integer (default: 10)" >&2
  exit 1
fi

if [[ ${#REMAINING_ARGS[@]} -gt 1 ]]; then
  echo "Warning: ignoring unknown arguments: ${REMAINING_ARGS[*]:1}" >&2
fi

RALPH_SANDBOX=1

require_jq
ensure_branch

echo "=== Ralph AFK loop: $ITERATIONS iterations (branch: $RALPH_BRANCH, model: $RALPH_MODEL) ==="

for ((i = 1; i <= ITERATIONS; i++)); do
  echo ""
  echo "=== Ralph iteration $i/$ITERATIONS ==="
  agent_exit=0
  run_ralph_iteration || agent_exit=$?
  if [[ $agent_exit -ne 0 ]]; then
    if is_user_abort_exit "$agent_exit"; then
      echo "Interrupted — stopping Ralph loop." >&2
      exit "$agent_exit"
    fi
    echo "Warning: iteration $i agent exited $agent_exit — continuing loop." >&2
  fi

  if is_complete; then
    echo "PRD complete, exiting."
    exit 0
  fi
done

echo "Max iterations reached. Review branch $RALPH_BRANCH and rerun."

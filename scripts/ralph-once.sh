#!/usr/bin/env bash
# HITL Ralph — run a single iteration and watch the output.
# Usage:
#   ./scripts/ralph-once.sh [--interactive|-i] [--model MODEL]
#   ./scripts/ralph-once.sh [--text|--no-stream] [--model MODEL]   # headless plain text
# Default: headless JSON stream (RALPH_STREAM=1)

set -euo pipefail

REMAINING_ARGS=()
source "$(cd "$(dirname "$0")" && pwd)/ralph-common.sh"

parse_ralph_args "$@"
RALPH_SANDBOX=""

ensure_branch

if [[ -n "$RALPH_INTERACTIVE" ]]; then
  echo "=== Ralph HITL interactive (branch: $RALPH_BRANCH, model: $RALPH_MODEL) ==="
else
  echo "=== Ralph HITL iteration (branch: $RALPH_BRANCH, model: $RALPH_MODEL) ==="
fi
run_ralph_iteration

if is_complete; then
  echo "All PRD items complete."
fi

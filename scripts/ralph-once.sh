#!/usr/bin/env bash
# HITL Ralph — run a single iteration and watch the output.
# Usage: ./scripts/ralph-once.sh [--model MODEL]

set -euo pipefail

REMAINING_ARGS=()
source "$(cd "$(dirname "$0")" && pwd)/ralph-common.sh"

parse_ralph_args "$@"
RALPH_SANDBOX=""

ensure_branch

echo "=== Ralph HITL iteration (branch: $RALPH_BRANCH, model: $RALPH_MODEL) ==="
run_ralph_iteration

if is_complete; then
  echo "PRD complete."
fi

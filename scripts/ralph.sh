#!/usr/bin/env bash
# AFK Ralph — run up to N autonomous iterations with sandbox enabled.
# Usage:
#   ./scripts/ralph.sh [iterations] [--model MODEL]
#   ./scripts/ralph.sh [iterations] [--text|--no-stream] [--model MODEL]
# Default: 10 iterations, headless JSON stream (RALPH_STREAM=1)

set -euo pipefail

REMAINING_ARGS=()
source "$(cd "$(dirname "$0")" && pwd)/ralph-common.sh"

# Parse optional iteration count (first numeric arg) and --model
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --interactive|-i)
      echo "Error: --interactive is only supported by ralph-once.sh (single HITL session)." >&2
      echo "Run: ./scripts/ralph-once.sh --interactive" >&2
      exit 1
      ;;
    --text|--no-stream)
      RALPH_STREAM=0
      shift
      ;;
    --model)
      RALPH_MODEL="$2"
      shift 2
      ;;
    --model=*)
      RALPH_MODEL="${1#*=}"
      shift
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done

ITERATIONS="${POSITIONAL[0]:-10}"
if ! [[ "$ITERATIONS" =~ ^[0-9]+$ ]] || [[ "$ITERATIONS" -lt 1 ]]; then
  echo "Usage: $0 [iterations] [--text|--no-stream] [--model MODEL]" >&2
  echo "  iterations must be a positive integer (default: 10)" >&2
  exit 1
fi

RALPH_SANDBOX=1

ensure_branch

echo "=== Ralph AFK loop: $ITERATIONS iterations (branch: $RALPH_BRANCH, model: $RALPH_MODEL) ==="

for ((i = 1; i <= ITERATIONS; i++)); do
  echo ""
  echo "=== Ralph iteration $i/$ITERATIONS ==="
  run_ralph_iteration

  if is_complete; then
    echo "PRD complete, exiting."
    exit 0
  fi
done

echo "Max iterations reached. Review branch $RALPH_BRANCH and rerun."

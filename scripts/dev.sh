#!/usr/bin/env bash
# Start/stop local dev: Docker (Redis), FastAPI backend, Tailwind CSS watcher.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DEV_DIR="$ROOT/.dev"
BACKEND_PID="$DEV_DIR/backend.pid"
TAILWIND_PID="$DEV_DIR/tailwind.pid"
BACKEND_PORT="${BACKEND_PORT:-8080}"
REDIS_PORT="${REDIS_PORT:-6379}"
BACKEND_LOG="$DEV_DIR/backend.log"
TAILWIND_LOG="$DEV_DIR/tailwind.log"

mkdir -p "$DEV_DIR"

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
else
  PYTHON="$(command -v python3)"
fi

log() {
  printf '%s\n' "$*"
}

ensure_docker() {
  if docker info >/dev/null 2>&1; then
    return 0
  fi

  log "Docker is not running — starting Docker Desktop..."
  if [[ "$(uname -s)" == "Darwin" ]]; then
    open -a Docker
  else
    log "Start Docker manually, then re-run: make dev" >&2
    exit 1
  fi

  local attempt=0
  while ! docker info >/dev/null 2>&1; do
    attempt=$((attempt + 1))
    if (( attempt > 60 )); then
      log "Timed out waiting for Docker to start." >&2
      exit 1
    fi
    sleep 2
  done
  log "Docker is ready."
}

kill_port() {
  local port=$1
  local pids
  pids="$(lsof -ti ":$port" 2>/dev/null || true)"
  if [[ -n "$pids" ]]; then
    log "Freeing port $port..."
    # shellcheck disable=SC2086
    kill $pids 2>/dev/null || true
    sleep 1
    pids="$(lsof -ti ":$port" 2>/dev/null || true)"
    if [[ -n "$pids" ]]; then
      # shellcheck disable=SC2086
      kill -9 $pids 2>/dev/null || true
    fi
  fi
}

stop_pid_file() {
  local pidfile=$1
  local name=$2
  if [[ ! -f "$pidfile" ]]; then
    return 0
  fi

  local pid
  pid="$(cat "$pidfile")"
  if kill -0 "$pid" 2>/dev/null; then
    log "Stopping $name (pid $pid)..."
    kill "$pid" 2>/dev/null || true
    local wait_attempt=0
    while kill -0 "$pid" 2>/dev/null; do
      wait_attempt=$((wait_attempt + 1))
      if (( wait_attempt > 10 )); then
        kill -9 "$pid" 2>/dev/null || true
        break
      fi
      sleep 0.5
    done
  fi
  rm -f "$pidfile"
}

remove_legacy_redis_container() {
  if ! docker ps -a --format '{{.Names}}' | grep -qx 'agentic-kapruka-redis'; then
    return 0
  fi
  if docker compose ps -q redis 2>/dev/null | grep -q .; then
    return 0
  fi
  log "Removing legacy standalone Redis container..."
  docker rm -f agentic-kapruka-redis >/dev/null 2>&1 || true
}

refresh_docker() {
  ensure_docker
  remove_legacy_redis_container

  if docker compose ps -q redis 2>/dev/null | grep -q .; then
    log "Refreshing Docker Compose services..."
    docker compose up -d --force-recreate --wait
  else
    log "Starting Docker Compose services..."
    docker compose up -d --wait
  fi
}

wait_redis() {
  log "Waiting for Redis on :$REDIS_PORT..."
  local attempt=0
  while true; do
    if command -v redis-cli >/dev/null 2>&1; then
      if redis-cli -h 127.0.0.1 -p "$REDIS_PORT" ping 2>/dev/null | grep -q PONG; then
        break
      fi
    elif docker compose exec -T redis redis-cli ping 2>/dev/null | grep -q PONG; then
      break
    fi
    attempt=$((attempt + 1))
    if (( attempt > 45 )); then
      log "Redis did not become ready on port $REDIS_PORT." >&2
      exit 1
    fi
    sleep 1
  done
  log "Redis is ready."
}

stop_dev_processes() {
  stop_pid_file "$BACKEND_PID" "backend"
  stop_pid_file "$TAILWIND_PID" "Tailwind watcher"
  kill_port "$BACKEND_PORT"
}

start_tailwind() {
  make -C "$ROOT" install-tailwind >/dev/null
  log "Starting Tailwind watcher..."
  nohup "$ROOT/bin/tailwindcss" \
    -i "$ROOT/static/css/input.css" \
    -o "$ROOT/static/css/app.css" \
    --watch \
    >"$TAILWIND_LOG" 2>&1 &
  echo $! >"$TAILWIND_PID"
}

start_backend() {
  log "Starting backend on http://127.0.0.1:$BACKEND_PORT ..."
  nohup "$PYTHON" -m uvicorn app.main:app \
    --reload \
    --host 127.0.0.1 \
    --port "$BACKEND_PORT" \
    >"$BACKEND_LOG" 2>&1 &
  echo $! >"$BACKEND_PID"
}

wait_backend() {
  # Any HTTP response means uvicorn is accepting connections (health may be 503
  # when Neo4j/Zep/MCP are degraded — that is fine for local dev startup).
  local attempt=0
  while ! curl -s -o /dev/null "http://127.0.0.1:$BACKEND_PORT/health" 2>/dev/null; do
    attempt=$((attempt + 1))
    if (( attempt > 30 )); then
      log "Backend did not start — see $BACKEND_LOG" >&2
      exit 1
    fi
    sleep 1
  done
}

cmd_start() {
  stop_dev_processes
  refresh_docker
  wait_redis
  make -C "$ROOT" css
  start_tailwind
  start_backend
  wait_backend

  log ""
  log "Dev environment is running."
  log "  Chat:     http://127.0.0.1:$BACKEND_PORT/chat"
  log "  Health:   http://127.0.0.1:$BACKEND_PORT/health"
  log "  Backend:  $BACKEND_LOG"
  log "  Tailwind: $TAILWIND_LOG"
  log "  Stop all: make stop-all"
}

cmd_stop() {
  stop_dev_processes
  if docker info >/dev/null 2>&1; then
    log "Running docker compose down..."
    docker compose down
  else
    log "Docker is not running — skipped compose down."
  fi
  log "All services stopped."
}

case "${1:-start}" in
  start) cmd_start ;;
  stop) cmd_stop ;;
  restart)
    stop_dev_processes
    cmd_start
    ;;
  *)
    log "Usage: $0 {start|stop|restart}" >&2
    exit 1
    ;;
esac

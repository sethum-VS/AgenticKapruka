# AgenticKapruka — Tailwind build + local dev orchestration
TAILWIND_BIN := bin/tailwindcss
INPUT_CSS := static/css/input.css
OUTPUT_CSS := static/css/app.css

BACKEND_PORT ?= 8080
REDIS_PORT ?= 6379

.PHONY: css css-watch install-tailwind clean-css
.PHONY: dev start restart stop-all stop status logs

css: $(OUTPUT_CSS)

$(OUTPUT_CSS): $(INPUT_CSS) tailwind.config.js $(TAILWIND_BIN)
	$(TAILWIND_BIN) -i $(INPUT_CSS) -o $(OUTPUT_CSS) --minify

css-watch: $(TAILWIND_BIN)
	$(TAILWIND_BIN) -i $(INPUT_CSS) -o $(OUTPUT_CSS) --watch

install-tailwind: $(TAILWIND_BIN)

$(TAILWIND_BIN):
	@./scripts/install-tailwind.sh

clean-css:
	rm -f $(OUTPUT_CSS)

# ── Local dev stack (Docker Redis + uvicorn backend + Tailwind watcher) ───────
# If already running, stops and restarts processes and refreshes Docker services.
dev start:
	@chmod +x scripts/dev.sh
	@BACKEND_PORT=$(BACKEND_PORT) REDIS_PORT=$(REDIS_PORT) ./scripts/dev.sh start

restart:
	@chmod +x scripts/dev.sh
	@BACKEND_PORT=$(BACKEND_PORT) REDIS_PORT=$(REDIS_PORT) ./scripts/dev.sh restart

stop-all stop:
	@chmod +x scripts/dev.sh
	@BACKEND_PORT=$(BACKEND_PORT) ./scripts/dev.sh stop

status:
	@echo "=== Docker ==="
	@docker compose ps 2>/dev/null || echo "Docker not running"
	@echo ""
	@echo "=== Ports ==="
	@lsof -nP -iTCP:$(BACKEND_PORT) -sTCP:LISTEN 2>/dev/null || echo "Backend (:$(BACKEND_PORT)): not listening"
	@lsof -nP -iTCP:$(REDIS_PORT) -sTCP:LISTEN 2>/dev/null || echo "Redis (:$(REDIS_PORT)): not listening"
	@echo ""
	@echo "=== PID files (.dev/) ==="
	@test -f .dev/backend.pid && echo "Backend pid: $$(cat .dev/backend.pid)" || echo "Backend: not started via make"
	@test -f .dev/tailwind.pid && echo "Tailwind pid: $$(cat .dev/tailwind.pid)" || echo "Tailwind: not started via make"

logs:
	@tail -n 50 -f .dev/backend.log .dev/tailwind.log

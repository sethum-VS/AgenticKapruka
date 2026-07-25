# Windows PowerShell Scripts

PowerShell ports of the bash scripts for local development on Windows.

## Local development

### `dev.ps1` — start/stop the stack

```powershell
.\scripts\dev.ps1 start
.\scripts\dev.ps1 stop
.\scripts\dev.ps1 restart
```

Starts Docker Compose (Redis), the FastAPI backend on port 8080, and the Tailwind watcher.

Optional env vars: `BACKEND_PORT` (8080), `REDIS_PORT` (6379), `APP_ENV`, `DEBUG_TRACE`, `LOG_LEVEL`.

### `install-tailwind.ps1` — Tailwind standalone CLI

```powershell
.\scripts\install-tailwind.ps1
```

Downloads the Windows Tailwind binary into `bin\tailwindcss.exe` (no Node.js required).

### `bootstrap_env.ps1` — generate `.env`

```powershell
.\scripts\bootstrap_env.ps1
.\scripts\bootstrap_env.ps1 -Force
```

## Prerequisites

- Docker Desktop running
- Python 3 (preferably project `.venv` at `.venv\Scripts\python.exe`)
- A filled-in `.env` (see `.env.example` or `bootstrap_env.ps1`)

## If PowerShell blocks scripts

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\dev.ps1 start
```

Or:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\dev.ps1 start
```

## Troubleshooting

- **Docker not running** — start Docker Desktop, then re-run `.\scripts\dev.ps1 start`
- **Python not found** — create `.venv` or put `python` on PATH
- **Port in use** — the script frees port 8080; or run `.\scripts\dev.ps1 stop` first
- **Logs** — `.dev\backend.log` and `.dev\tailwind.log` (populated when the backend is started via `.\scripts\dev.ps1 start`). Tail with `.\scripts\dev.ps1 logs` or `make logs`.

## Bash / Make (macOS, Linux, Git Bash)

The original `.sh` scripts and `Makefile` targets (`make dev`, `make stop-all`, …) remain for Unix environments. Production/deploy scripts (`deploy_cloud_run.sh`, `setup_github_cicd.sh`, `verify_production_prerequisites.sh`) are still bash; use Git Bash or WSL for those.

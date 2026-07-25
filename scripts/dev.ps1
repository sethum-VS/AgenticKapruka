# Start/stop local dev: Docker (Redis), FastAPI backend, Tailwind CSS watcher.
# Windows PowerShell port of scripts/dev.sh
#
# Usage: .\scripts\dev.ps1 [start|stop|restart|logs]

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet("start", "stop", "restart", "logs")]
    [string]$Command = "start"
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$DevDir = Join-Path $Root ".dev"
$BackendPidFile = Join-Path $DevDir "backend.pid"
$TailwindPidFile = Join-Path $DevDir "tailwind.pid"
$BackendPort = if ($env:BACKEND_PORT) { [int]$env:BACKEND_PORT } else { 8080 }
$RedisPort = if ($env:REDIS_PORT) { [int]$env:REDIS_PORT } else { 6379 }
$BackendLog = Join-Path $DevDir "backend.log"
$TailwindLog = Join-Path $DevDir "tailwind.log"

if (-not (Test-Path $DevDir)) {
    New-Item -ItemType Directory -Path $DevDir -Force | Out-Null
}

function Resolve-Python {
    $venvPython = Join-Path $Root ".venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        return $venvPython
    }
    $venvPythonUnix = Join-Path $Root ".venv\bin\python"
    if (Test-Path $venvPythonUnix) {
        return $venvPythonUnix
    }
    foreach ($name in @("python", "python3", "py")) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($cmd) {
            return $cmd.Source
        }
    }
    throw "Python not found. Create .venv or ensure python is on PATH."
}

$Python = Resolve-Python

function Write-Log {
    param([string]$Message)
    Write-Host $Message
}

function Start-CmdLogged {
    param(
        [string]$CommandLine,
        [string]$LogFile,
        [string]$WorkingDirectory
    )

    # cmd redirect keeps a single combined log; PYTHONUNBUFFERED ensures prompt flush
    # so Get-Content -Wait / make logs can tail while the process runs.
    $bat = Join-Path $DevDir ("run-" + [guid]::NewGuid().ToString("N") + ".cmd")
    @(
        "@echo off"
        "cd /d `"$WorkingDirectory`""
        "set PYTHONUNBUFFERED=1"
        "$CommandLine > `"$LogFile`" 2>&1"
    ) | Set-Content -Path $bat -Encoding ASCII

    $proc = Start-Process -FilePath $bat `
        -WorkingDirectory $WorkingDirectory `
        -PassThru `
        -WindowStyle Hidden

    return @{ Process = $proc; BatFile = $bat }
}

function Test-DockerRunning {
    # docker info often writes warnings to stderr; with $ErrorActionPreference=Stop
    # those become terminating errors. Probe via cmd so stderr is not PowerShell errors.
    cmd /c "docker info >NUL 2>&1"
    return ($LASTEXITCODE -eq 0)
}

function Ensure-Docker {
    if (Test-DockerRunning) {
        return
    }

    Write-Log "Docker is not running - starting Docker Desktop..."
    $dockerDesktop = @(
        "$env:LOCALAPPDATA\Programs\DockerDesktop\Docker Desktop.exe",
        "${env:ProgramFiles}\Docker\Docker\Docker Desktop.exe",
        "${env:ProgramFiles(x86)}\Docker\Docker\Docker Desktop.exe",
        "$env:LOCALAPPDATA\Docker\Docker Desktop.exe"
    ) | Where-Object { Test-Path $_ } | Select-Object -First 1

    if (-not $dockerDesktop) {
        # Fall back: locate beside docker.exe
        $dockerCmd = Get-Command docker -ErrorAction SilentlyContinue
        if ($dockerCmd) {
            $candidate = Join-Path (Split-Path (Split-Path $dockerCmd.Source)) "Docker Desktop.exe"
            if (Test-Path $candidate) { $dockerDesktop = $candidate }
        }
    }

    if ($dockerDesktop) {
        Start-Process $dockerDesktop | Out-Null
    }
    else {
        Write-Log "Start Docker Desktop manually, then re-run: .\scripts\dev.ps1 start"
        exit 1
    }

    $attempt = 0
    while (-not (Test-DockerRunning)) {
        $attempt++
        if ($attempt -gt 60) {
            Write-Log "Timed out waiting for Docker to start."
            exit 1
        }
        Start-Sleep -Seconds 2
    }
    Write-Log "Docker is ready."
}

function Stop-PortListeners {
    param([int]$Port)

    try {
        $conns = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    }
    catch {
        $conns = $null
    }

    if (-not $conns) {
        return
    }

    $procIds = $conns | Select-Object -ExpandProperty OwningProcess -Unique
    Write-Log "Freeing port $Port..."
    foreach ($procId in $procIds) {
        if ($procId -and $procId -ne 0) {
            Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
        }
    }
    Start-Sleep -Seconds 1

    try {
        $conns = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    }
    catch {
        $conns = $null
    }
    if ($conns) {
        $procIds = $conns | Select-Object -ExpandProperty OwningProcess -Unique
        foreach ($procId in $procIds) {
            if ($procId -and $procId -ne 0) {
                Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
            }
        }
    }
}

function Stop-PidFile {
    param(
        [string]$PidFile,
        [string]$Name
    )

    if (-not (Test-Path $PidFile)) {
        return
    }

    $procIdText = (Get-Content $PidFile -Raw).Trim()
    $procId = 0
    if (-not [int]::TryParse($procIdText, [ref]$procId)) {
        Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
        return
    }

    $proc = Get-Process -Id $procId -ErrorAction SilentlyContinue
    if ($proc) {
        Write-Log "Stopping $Name (pid $procId)..."
        # Kill process tree (cmd wrappers spawn children)
        try {
            & taskkill.exe /PID $procId /T /F 1>$null 2>$null
        }
        catch {
            Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
        }
        $waitAttempt = 0
        while (Get-Process -Id $procId -ErrorAction SilentlyContinue) {
            $waitAttempt++
            if ($waitAttempt -gt 10) {
                Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
                break
            }
            Start-Sleep -Milliseconds 500
        }
    }
    Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
}

function Remove-LegacyRedisContainer {
    $names = @(cmd /c "docker ps -a --format {{.Names}} 2>NUL")
    if ($LASTEXITCODE -ne 0) {
        return
    }
    if ($names -notcontains "agentic-kapruka-redis") {
        return
    }

    $composeRedis = @(cmd /c "docker compose ps -q redis 2>NUL")
    if ($composeRedis | Where-Object { $_.Trim() }) {
        return
    }

    Write-Log "Removing legacy standalone Redis container..."
    cmd /c "docker rm -f agentic-kapruka-redis >NUL 2>&1" | Out-Null
}

function Refresh-Docker {
    Ensure-Docker
    Remove-LegacyRedisContainer

    $redisId = @(cmd /c "docker compose ps -q redis 2>NUL")
    if ($redisId | Where-Object { $_.Trim() }) {
        Write-Log "Refreshing Docker Compose services..."
        docker compose up -d --force-recreate --wait
    }
    else {
        Write-Log "Starting Docker Compose services..."
        docker compose up -d --wait
    }
    if ($LASTEXITCODE -ne 0) {
        throw "docker compose up failed"
    }
}

function Wait-Redis {
    Write-Log "Waiting for Redis on :$RedisPort..."
    $attempt = 0
    while ($true) {
        $ready = $false
        $redisCli = Get-Command redis-cli -ErrorAction SilentlyContinue
        if ($redisCli) {
            $pong = cmd /c "redis-cli -h 127.0.0.1 -p $RedisPort ping 2>NUL"
            if ($pong -match "PONG") {
                $ready = $true
            }
        }
        else {
            $pong = cmd /c "docker compose exec -T redis redis-cli ping 2>NUL"
            if ($pong -match "PONG") {
                $ready = $true
            }
        }

        if ($ready) {
            break
        }

        $attempt++
        if ($attempt -gt 45) {
            Write-Log "Redis did not become ready on port $RedisPort."
            exit 1
        }
        Start-Sleep -Seconds 1
    }
    Write-Log "Redis is ready."
}

function Stop-DevProcesses {
    Stop-PidFile -PidFile $BackendPidFile -Name "backend"
    Stop-PidFile -PidFile $TailwindPidFile -Name "Tailwind watcher"
    Stop-PortListeners -Port $BackendPort
}

function Invoke-CssBuild {
    $tailwind = Join-Path $Root "bin\tailwindcss.exe"
    if (-not (Test-Path $tailwind)) {
        & (Join-Path $PSScriptRoot "install-tailwind.ps1")
    }
    $inputCss = Join-Path $Root "static\css\input.css"
    $outputCss = Join-Path $Root "static\css\app.css"
    & $tailwind -i $inputCss -o $outputCss --minify
    if ($LASTEXITCODE -ne 0) {
        throw "Tailwind CSS build failed"
    }
}

function Start-Tailwind {
    & (Join-Path $PSScriptRoot "install-tailwind.ps1")
    Write-Log "Starting Tailwind watcher..."

    $tailwind = Join-Path $Root "bin\tailwindcss.exe"
    $inputCss = Join-Path $Root "static\css\input.css"
    $outputCss = Join-Path $Root "static\css\app.css"

    $cmdLine = "`"$tailwind`" -i `"$inputCss`" -o `"$outputCss`" --watch"
    $started = Start-CmdLogged -CommandLine $cmdLine -LogFile $TailwindLog -WorkingDirectory $Root
    Set-Content -Path $TailwindPidFile -Value $started.Process.Id -NoNewline
}

function Start-Backend {
    Write-Log "Starting backend on http://127.0.0.1:$BackendPort ..."

    if (-not $env:APP_ENV) { $env:APP_ENV = "development" }
    if (-not $env:DEBUG_TRACE) { $env:DEBUG_TRACE = "1" }
    if (-not $env:LOG_LEVEL) { $env:LOG_LEVEL = "INFO" }
    $env:PYTHONUNBUFFERED = "1"

    $uvicornLogLevel = $env:LOG_LEVEL.ToLowerInvariant()
    $cmdLine = "`"$Python`" -m uvicorn app.main:app --reload --host 127.0.0.1 --port $BackendPort --log-level $uvicornLogLevel"
    $started = Start-CmdLogged -CommandLine $cmdLine -LogFile $BackendLog -WorkingDirectory $Root
    Set-Content -Path $BackendPidFile -Value $started.Process.Id -NoNewline
}

function Invoke-CmdLogs {
    Write-Log "Tailing $BackendLog and $TailwindLog (Ctrl+C to stop)..."
    $files = @()
    if (Test-Path $BackendLog) { $files += $BackendLog } else { Write-Log "Missing: $BackendLog (start via .\scripts\dev.ps1 start)" }
    if (Test-Path $TailwindLog) { $files += $TailwindLog }
    if ($files.Count -eq 0) {
        Write-Log "No log files found under .dev\"
        exit 1
    }
    Get-Content -Path $files -Wait -Tail 50
}

function Test-HttpReachable {
    param([string]$Url)
    # Use cmd so curl progress/warnings on stderr do not become terminating errors
    # under $ErrorActionPreference = "Stop".
    $code = cmd /c "curl.exe -s -o NUL -w %{http_code} --connect-timeout 2 --max-time 5 `"$Url`" 2>NUL"
    if ($LASTEXITCODE -eq 0 -and $code -and $code -ne "000") {
        return $true
    }
    return $false
}

function Wait-Backend {
    # Any HTTP response means uvicorn is accepting connections (health may be 503
    # when Neo4j/Zep/MCP are degraded - that is fine for local dev startup).
    $attempt = 0
    $url = "http://127.0.0.1:$BackendPort/health"
    while (-not (Test-HttpReachable -Url $url)) {
        $attempt++
        if ($attempt -gt 90) {
            Write-Log "Backend did not start - see $BackendLog"
            exit 1
        }
        Start-Sleep -Seconds 1
    }
}

function Test-BackendHealthy {
    return (Test-HttpReachable -Url "http://127.0.0.1:$BackendPort/health")
}

function Invoke-CmdStart {
    if (Test-Path $BackendPidFile) {
        $existingText = (Get-Content $BackendPidFile -Raw).Trim()
        $existingPid = 0
        if ([int]::TryParse($existingText, [ref]$existingPid)) {
            if ((Get-Process -Id $existingPid -ErrorAction SilentlyContinue) -and (Test-BackendHealthy)) {
                Write-Log "Backend already running on http://127.0.0.1:$BackendPort - use '.\scripts\dev.ps1 restart' to reload."
                exit 0
            }
        }
    }

    Stop-DevProcesses
    Refresh-Docker
    Wait-Redis
    Invoke-CssBuild
    Start-Tailwind
    Start-Backend
    Wait-Backend

    $debugTrace = $env:DEBUG_TRACE
    $logLevel = $env:LOG_LEVEL

    Write-Log ""
    Write-Log "Dev environment is running."
    Write-Log "  Chat:     http://127.0.0.1:$BackendPort/chat"
    Write-Log "  Health:   http://127.0.0.1:$BackendPort/health"
    Write-Log "  Backend:  $BackendLog"
    Write-Log "  Tailwind: $TailwindLog"
    Write-Log "  Trace:    DEBUG_TRACE=$debugTrace LOG_LEVEL=$logLevel"
    Write-Log "  Stop all: .\scripts\dev.ps1 stop"

    try {
        $healthBody = cmd /c "curl.exe -s --connect-timeout 5 --max-time 8 http://127.0.0.1:$BackendPort/health 2>NUL"
        if ($healthBody) {
            $py = @'
import sys, json
d = json.load(sys.stdin)
v = d.get("services", {}).get("neo4j_graphrag", "unknown")
print(v.get("status", v) if isinstance(v, dict) else v)
'@
            $neo4jStatus = $healthBody | & $Python -c $py 2>$null
            if ($neo4jStatus -and $neo4jStatus.Trim() -ne "healthy" -and $neo4jStatus.Trim() -ne "up") {
                Write-Log ""
                Write-Log "  WARNING: Neo4j GraphRAG is '$($neo4jStatus.Trim())' - carousel quality may degrade."
                Write-Log "     Run: python scripts\bootstrap_neo4j.py"
            }
        }
    }
    catch {
        # Non-fatal status check
    }
}

function Invoke-CmdStop {
    Stop-DevProcesses
    if (Test-DockerRunning) {
        Write-Log "Running docker compose down..."
        docker compose down
    }
    else {
        Write-Log "Docker is not running - skipped compose down."
    }
    Write-Log "All services stopped."
}

switch ($Command) {
    "start" { Invoke-CmdStart }
    "stop" { Invoke-CmdStop }
    "restart" {
        Stop-DevProcesses
        Invoke-CmdStart
    }
    "logs" { Invoke-CmdLogs }
}

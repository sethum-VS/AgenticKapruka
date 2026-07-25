# Sync local .env values into GitHub repository secrets for Heroku CI/CD deploy.
# Usage: .\scripts\setup_github_heroku_secrets.ps1
# Requires: gh auth login (repo secret write access)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$EnvFile = Join-Path $Root ".env"

if (-not (Test-Path $EnvFile)) {
    Write-Error ".env not found at $EnvFile"
}

$SecretKeys = @(
    "REDIS_URL",
    "NEO4J_URI",
    "NEO4J_USER",
    "NEO4J_PASSWORD",
    "ZEP_API_KEY",
    "NVIDIA_API_KEY",
    "NVIDIA_API_KEY_BACKUP",
    "NVIDIA_BASE_URL",
    "NVIDIA_LLM_MODEL",
    "NVIDIA_EMBEDDING_MODEL",
    "NVIDIA_RATE_LIMIT_RPM",
    "NVIDIA_VECTOR_THRESHOLD",
    "KAPRUKA_MCP_URL",
    "RERANKER_THRESHOLD",
    "SESSION_SECRET"
)

$Values = @{}
Get-Content $EnvFile | ForEach-Object {
    $line = $_.Trim()
    if ($line -eq "" -or $line.StartsWith("#")) { return }
    $idx = $line.IndexOf("=")
    if ($idx -lt 1) { return }
    $key = $line.Substring(0, $idx).Trim()
    $value = $line.Substring($idx + 1).Trim()
    if ($value.StartsWith('"') -and $value.EndsWith('"')) {
        $value = $value.Substring(1, $value.Length - 2)
    }
    $Values[$key] = $value
}

foreach ($key in $SecretKeys) {
    if (-not $Values.ContainsKey($key) -or [string]::IsNullOrWhiteSpace($Values[$key])) {
        Write-Warning "Skipping $key (missing or empty in .env)"
        continue
    }
    Write-Host "Setting GitHub secret: $key"
    $Values[$key] | gh secret set $key
}

Write-Host "Done. HEROKU_API_KEY and HEROKU_APP_NAME must already exist in GitHub secrets."

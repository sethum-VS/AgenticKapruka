# Download Tailwind CSS standalone CLI for Windows (no Node.js required).
# Mirrors scripts/install-tailwind.sh
#
# Usage: .\scripts\install-tailwind.ps1

$ErrorActionPreference = "Stop"

$Version = "v3.4.17"
$RootDir = Split-Path -Parent $PSScriptRoot
$BinDir = Join-Path $RootDir "bin"
$Target = Join-Path $BinDir "tailwindcss.exe"
$StampFile = Join-Path $BinDir ".tailwindcss-platform"

if (-not (Test-Path $BinDir)) {
    New-Item -ItemType Directory -Path $BinDir -Force | Out-Null
}

$arch = $env:PROCESSOR_ARCHITECTURE
switch ($arch) {
    "AMD64" { $archTag = "x86_64" }
    "ARM64" { $archTag = "arm64" }
    default {
        Write-Error "Unsupported architecture: $arch"
        exit 1
    }
}

$PlatformTag = "windows-$archTag"
$StampExpected = "${Version}:${PlatformTag}"

switch ($PlatformTag) {
    "windows-x86_64" { $Asset = "tailwindcss-windows-x64.exe" }
    "windows-arm64"  { $Asset = "tailwindcss-windows-arm64.exe" }
    default {
        Write-Error "Unsupported platform: $PlatformTag"
        exit 1
    }
}

if ((Test-Path $Target) -and (Test-Path $StampFile)) {
    $stamp = (Get-Content $StampFile -Raw).Trim()
    if ($stamp -eq $StampExpected) {
        Write-Host "Tailwind CSS $Version already installed for ${PlatformTag}: $Target"
        exit 0
    }
}

if (Test-Path $Target) {
    Write-Host "Re-downloading Tailwind CSS $Version for $PlatformTag (version or platform mismatch)..."
}

$Url = "https://github.com/tailwindlabs/tailwindcss/releases/download/${Version}/${Asset}"
Write-Host "Downloading Tailwind CSS $Version ($Asset)..."

Invoke-WebRequest -Uri $Url -OutFile $Target -UseBasicParsing

Set-Content -Path $StampFile -Value $StampExpected -NoNewline
Write-Host "Installed $Target"

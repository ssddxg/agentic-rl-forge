[CmdletBinding()]
param(
    [string]$BindHost = "127.0.0.1",
    [ValidateRange(1, 65535)]
    [int]$Port = 7860,
    [string]$DataDir,
    [switch]$NoOpen,
    [switch]$AllowNetwork
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

try {
    $projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
    $arf = Join-Path $projectRoot ".venv\Scripts\arf.exe"
    $python = Join-Path $projectRoot ".venv\Scripts\python.exe"

    if ($BindHost -notin @("127.0.0.1", "localhost", "::1") -and -not $AllowNetwork) {
        throw "Non-local access requires -AllowNetwork. Keep the default address for local use."
    }

    $needsSetup = -not (Test-Path -LiteralPath $arf -PathType Leaf) -or
        -not (Test-Path -LiteralPath $python -PathType Leaf)
    if (-not $needsSetup) {
        $probe = "import sys, docx, fastapi, multipart, pypdf; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else 1)"
        & $python "-c" $probe *> $null
        $dependenciesReady = $LASTEXITCODE -eq 0
        $staticDirectory = Join-Path $projectRoot "src\agentic_rl_forge\studio\static"
        $requiredAssets = @("index.html", "styles.css", "app.js", "icon.svg")
        $assetsReady = @(
            $requiredAssets | Where-Object {
                -not (Test-Path -LiteralPath (Join-Path $staticDirectory $_) -PathType Leaf)
            }
        ).Count -eq 0
        $needsSetup = -not $dependenciesReady -or -not $assetsReady
    }

    if ($needsSetup) {
        Write-Host "First run: installing AgenticRLForge Studio..." -ForegroundColor Cyan
        & (Join-Path $PSScriptRoot "setup.ps1") -RuntimeOnly
        if ($LASTEXITCODE -ne 0) {
            throw "Automatic installation failed with exit code $LASTEXITCODE."
        }
    }

    $arguments = @("studio", "--host", $BindHost, "--port", $Port.ToString())
    if ($NoOpen) {
        $arguments += "--no-open"
    }
    if ($AllowNetwork) {
        $arguments += "--allow-network"
    }
    if (-not [string]::IsNullOrWhiteSpace($DataDir)) {
        $arguments += @("--data-dir", $DataDir)
    }

    Write-Host "Starting AgenticRLForge. Keep this window open while using Studio." -ForegroundColor Green
    & $arf @arguments
    exit $LASTEXITCODE
}
catch {
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

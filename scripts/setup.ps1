[CmdletBinding()]
param(
    [switch]$RuntimeOnly,
    [switch]$All,
    [switch]$Recreate,
    [switch]$SkipValidation
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Step {
    param([Parameter(Mandatory)][string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Invoke-Native {
    param(
        [Parameter(Mandatory)][string]$Description,
        [Parameter(Mandatory)][string]$FilePath,
        [Parameter()][string[]]$Arguments = @()
    )

    Write-Step $Description
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE. Review the command output above."
    }
}

function Test-SupportedPython {
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [Parameter()][string[]]$PrefixArguments = @()
    )

    try {
        & $FilePath @PrefixArguments "-c" "import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else 1)" 2>$null
        return $LASTEXITCODE -eq 0
    }
    catch {
        return $false
    }
}

function Test-SupportedVenv {
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [Parameter(Mandatory)][string]$VenvDirectory
    )

    $probe = @"
import os
import sys
expected = os.path.normcase(os.path.realpath(sys.argv[1]))
actual = os.path.normcase(os.path.realpath(sys.prefix))
supported = (3, 10) <= sys.version_info[:2] <= (3, 12)
raise SystemExit(0 if supported and sys.prefix != sys.base_prefix and actual == expected else 1)
"@
    try {
        & $FilePath "-c" $probe $VenvDirectory 2>$null
        return $LASTEXITCODE -eq 0
    }
    catch {
        return $false
    }
}

function Get-PythonVersion {
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [Parameter()][string[]]$PrefixArguments = @()
    )

    $version = & $FilePath @PrefixArguments "-c" "import platform; print(platform.python_version())"
    if ($LASTEXITCODE -ne 0) {
        throw "Could not determine the Python version for $FilePath."
    }
    return ($version | Select-Object -Last 1).ToString().Trim()
}

function Find-SupportedPython {
    $candidates = @(
        [pscustomobject]@{ Name = "py"; Prefix = @("-3.12") },
        [pscustomobject]@{ Name = "py"; Prefix = @("-3.11") },
        [pscustomobject]@{ Name = "py"; Prefix = @("-3.10") },
        [pscustomobject]@{ Name = "python3.12"; Prefix = @() },
        [pscustomobject]@{ Name = "python3.11"; Prefix = @() },
        [pscustomobject]@{ Name = "python3.10"; Prefix = @() },
        [pscustomobject]@{ Name = "python3"; Prefix = @() },
        [pscustomobject]@{ Name = "python"; Prefix = @() }
    )

    foreach ($candidate in $candidates) {
        $command = Get-Command $candidate.Name -CommandType Application -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($null -eq $command) {
            continue
        }
        if (Test-SupportedPython -FilePath $command.Source -PrefixArguments $candidate.Prefix) {
            return [pscustomobject]@{
                FilePath = $command.Source
                Prefix = [string[]]$candidate.Prefix
            }
        }
    }
    return $null
}

function Find-VenvCommands {
    param([Parameter(Mandatory)][string]$VenvDirectory)

    $windowsPython = Join-Path $VenvDirectory "Scripts\python.exe"
    if (Test-Path -LiteralPath $windowsPython -PathType Leaf) {
        return [pscustomobject]@{
            Python = $windowsPython
            Arf = Join-Path $VenvDirectory "Scripts\arf.exe"
            Activate = Join-Path $VenvDirectory "Scripts\Activate.ps1"
        }
    }

    $unixPython = Join-Path $VenvDirectory "bin/python"
    if (Test-Path -LiteralPath $unixPython -PathType Leaf) {
        return [pscustomobject]@{
            Python = $unixPython
            Arf = Join-Path $VenvDirectory "bin/arf"
            Activate = Join-Path $VenvDirectory "bin/Activate.ps1"
        }
    }
    return $null
}

try {
    if ($RuntimeOnly -and $All) {
        throw "-RuntimeOnly and -All cannot be used together."
    }

    $projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
    $pyproject = Join-Path $projectRoot "pyproject.toml"
    $venvDirectory = Join-Path $projectRoot ".venv"

    if (-not (Test-Path -LiteralPath $pyproject -PathType Leaf)) {
        throw "pyproject.toml was not found at $projectRoot. Use a complete project checkout."
    }

    if (Test-Path -LiteralPath $venvDirectory) {
        $venvItem = Get-Item -LiteralPath $venvDirectory -Force
        if (-not $venvItem.PSIsContainer) {
            throw "$venvDirectory exists but is not a directory. Move it aside and try again."
        }
        if (($venvItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "$venvDirectory is a reparse point. Refusing to clear or replace a linked environment."
        }
    }

    $venvCommands = Find-VenvCommands -VenvDirectory $venvDirectory
    $reuseVenv = -not $Recreate -and $null -ne $venvCommands -and
        (Test-SupportedVenv -FilePath $venvCommands.Python -VenvDirectory $venvDirectory)

    if ($reuseVenv) {
        $version = Get-PythonVersion -FilePath $venvCommands.Python
        Write-Step "Reusing .venv with Python $version"
    }
    else {
        if (Test-Path -LiteralPath $venvDirectory -PathType Container) {
            Write-Warning "The existing .venv is incomplete, unsupported, or was explicitly marked for recreation."
        }

        $basePython = Find-SupportedPython
        if ($null -eq $basePython) {
            throw "Python 3.10-3.12 was not found. Install Python 3.12 from python.org, enable the py launcher, and rerun this script."
        }

        $version = Get-PythonVersion -FilePath $basePython.FilePath -PrefixArguments $basePython.Prefix
        $createArguments = @($basePython.Prefix) + @("-m", "venv", "--clear", $venvDirectory)
        Invoke-Native -Description "Creating .venv with Python $version" `
            -FilePath $basePython.FilePath -Arguments $createArguments

        $venvCommands = Find-VenvCommands -VenvDirectory $venvDirectory
        if ($null -eq $venvCommands) {
            throw "The environment was created, but its Python executable could not be found."
        }
        if (-not (Test-SupportedVenv -FilePath $venvCommands.Python -VenvDirectory $venvDirectory)) {
            throw "The new environment is not a valid project-local Python 3.10-3.12 virtual environment."
        }
    }

    if ($All) {
        $extras = "dev,studio,research,data,object-store,signing"
    }
    else {
        # The normal installation is intentionally the end-user Studio.  RuntimeOnly is kept as
        # a backwards-compatible spelling for setup automation and older documentation.
        $extras = "studio"
    }
    $installSpec = "${projectRoot}[$extras]"

    Push-Location -LiteralPath $projectRoot
    try {
        Invoke-Native -Description "Upgrading pip" -FilePath $venvCommands.Python `
            -Arguments @("-m", "pip", "install", "--upgrade", "pip")
        Invoke-Native -Description "Installing AgenticRLForge with extras: $extras" `
            -FilePath $venvCommands.Python `
            -Arguments @("-m", "pip", "install", "--editable", $installSpec)
        Invoke-Native -Description "Checking installed dependency consistency" `
            -FilePath $venvCommands.Python -Arguments @("-m", "pip", "check")

        if (-not $SkipValidation) {
            if (-not (Test-Path -LiteralPath $venvCommands.Arf -PathType Leaf)) {
                throw "The arf command was not installed at $($venvCommands.Arf)."
            }
            Invoke-Native -Description "Running arf doctor" -FilePath $venvCommands.Arf `
                -Arguments @("doctor", "--profile", "server", "--project", $projectRoot, "--strict")
            Invoke-Native -Description "Validating the local Studio dependencies" `
                -FilePath $venvCommands.Python `
                -Arguments @("-c", "import docx, fastapi, multipart, pypdf")
            $staticDirectory = Join-Path $projectRoot "src\agentic_rl_forge\studio\static"
            $requiredAssets = @("index.html", "styles.css", "app.js", "icon.svg")
            $missingAssets = @(
                $requiredAssets | Where-Object {
                    -not (Test-Path -LiteralPath (Join-Path $staticDirectory $_) -PathType Leaf)
                }
            )
            if ($missingAssets.Count -gt 0) {
                throw "Studio web assets are missing: $($missingAssets -join ', ')."
            }
            Invoke-Native -Description "Running the offline arf demo" -FilePath $venvCommands.Arf `
                -Arguments @("demo")
        }
    }
    finally {
        Pop-Location
    }

    Write-Host ""
    Write-Host "Setup complete." -ForegroundColor Green
    Write-Host "Activate it with: & `"$($venvCommands.Activate)`""
    Write-Host "Open the local app with: & `"$(Join-Path $PSScriptRoot 'start-studio.ps1')`""
    Write-Host "Run all local quality checks with: & `"$(Join-Path $PSScriptRoot 'check.ps1')`""
}
catch {
    Write-Host ""
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
